/*
Copyright (c) 2015, Intel Corporation
Copyright (c) 2025, Christian Asch

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
* Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*******************************************************************

NAME:    PIC

PURPOSE: This program tests the efficiency with which a cloud of
         charged particles can be moved through a spatially fixed
         collection of charges located at the vertices of a square
         equi-spaced grid. It is a proxy for a component of a
         particle-in-cell method

USAGE:   <progname> -s <#simulation steps> -g <grid size> -t <#particles> \
                    -p <horizontal velocity> -v <vertical velocity>    \
                    <init mode> <init parameters>

         The output consists of diagnostics to make sure the
         algorithm worked, and of timing statistics.

HISTORY: - Written by Evangelos Georganas, August 2015.
         - RvdW: Refactored to make the code PRK conforming, December 2015
         - Ported to Rust by Christian Asch, March 2025

**********************************************************************************/

use clap::{Parser, Subcommand};
use mpi;
use mpi::collective::SystemOperation;
use mpi::datatype::UserDatatype;
use mpi::request::RequestCollection;
use mpi::traits::*;
use std::f64::consts::PI;
use std::time::Instant;

const Q: f64 = 1.0;
const DT: f64 = 1.0;
const MASS_INV: f64 = 1.0;
const REL_X: f64 = 0.5;
const REL_Y: f64 = 0.5;
const EPSILON: f64 = 0.000001;
const ROOT_RANK: i32 = 0;

/// Particle initialization mode
#[derive(Subcommand, Debug, Clone)]
enum InitStyle {
    Geometric {
        /// Attenuation Factor
        #[arg(short, long, value_name = "rho")]
        attenuation_factor: f64,
    },
    Sinusoidal,
    Linear {
        /// Negative slope
        #[arg(short, long, value_name = "alpha")]
        negative_slope: f64,
        /// Constant offset
        #[arg(short, long, value_name = "beta")]
        constant_offset: f64,
    },
    Patch {
        #[arg(short, long)]
        xleft: u64,
        #[arg(short, long)]
        xright: u64,
        #[arg(short, long)]
        ybottom: u64,
        #[arg(short, long)]
        ytop: u64,
    },
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    /// Total number of simulation steps
    pub iterations: u64,
    #[arg(short, long, value_name = "L")]
    /// Dimension of grid in cells
    pub grid_size: u64,
    /// Total number of generated particles
    #[arg(short, long)]
    pub total_particles: u64,
    /// Initial horizontal velocity of particles
    #[arg(short, long, value_name = "k")]
    pub particle_charge_semi_increment: u64,
    /// Initial vertical velocity of particles
    #[arg(short, long, value_name = "m")]
    pub vertical_particle_velocity: u64,
    #[command(subcommand)]
    init_style: InitStyle,
}

#[derive(Debug)]
struct BoundingBox {
    left: u64,
    right: u64,
    bottom: u64,
    top: u64,
}

fn bad_patch(patch: &BoundingBox, patch_contain: &BoundingBox) -> bool {
    if patch.left >= patch.right || patch.bottom >= patch.top {
        return true;
    }
    if patch.left < patch_contain.left
        || patch.right > patch_contain.right
        || patch.bottom < patch_contain.bottom
        || patch.top > patch_contain.top
    {
        return true;
    }
    return false;
}

#[derive(Default, Debug, Clone, Copy)]
struct Particle {
    x: f64,
    y: f64,
    v_x: f64,
    v_y: f64,
    q: f64,
    x0: f64,
    y0: f64,
    k: f64,
    m: f64,
    id: f64,
}

unsafe impl Equivalence for Particle {
    type Out = UserDatatype;

    fn equivalent_datatype() -> Self::Out {
        mpi::datatype::UncommittedUserDatatype::contiguous(10, &f64::equivalent_datatype()).commit()
    }
}

#[derive(Debug)]
struct Grid<T> {
    data: Vec<T>,
    _cols: usize,
    rows: usize,
}

impl<T: Copy> Grid<T> {
    fn new(default_value: T, cols: usize, rows: usize) -> Self {
        Grid::<T> {
            data: vec![default_value; cols * rows],
            _cols: cols,
            rows,
        }
    }

    fn get(&self, col_idx: usize, row_idx: usize) -> T {
        let index = col_idx * self.rows + row_idx;
        self.data[index]
    }

    fn set(&mut self, col_idx: usize, row_idx: usize, value: T) {
        let index = col_idx * self.rows + row_idx;
        self.data[index] = value;
    }
}

fn initialize_grid(tile: &BoundingBox) -> Grid<f64> {
    let n_cols = tile.right - tile.left + 1;
    let n_rows = tile.top - tile.bottom + 1;
    let mut grid = Grid::<f64>::new(0f64, n_cols as usize, n_rows as usize);
    for col_idx in tile.left..(tile.right + 1) {
        for row_idx in tile.bottom..(tile.top + 1) {
            let value = match col_idx % 2 {
                0 => Q,
                _ => -Q,
            };
            grid.set(
                (col_idx - tile.left) as usize,
                (row_idx - tile.bottom) as usize,
                value,
            );
        }
    }
    grid
}

fn finalize_distribution(particles: &mut [Particle], comm: &mpi::topology::SimpleCommunicator) {
    let mut total_size: usize = 42;
    comm.scan_into(&particles.len(), &mut total_size, SystemOperation::sum());
    let mut id = total_size - particles.len() + 1;
    for particle in particles {
        let x_coord = particle.x;
        let y_coord = particle.y;
        let rel_x = x_coord % 1.0;
        let rel_y = y_coord % 1.0;
        let x = x_coord as u64;
        let r1_sq = rel_y * rel_y + rel_x * rel_x;
        let r2_sq = rel_y * rel_y + (1.0 - rel_x) * (1.0 - rel_x);
        let cos_theta = rel_x / r1_sq.sqrt();
        let cos_phi = (1.0 - rel_x) / r2_sq.sqrt();
        let base_charge = 1.0 / ((DT * DT) * Q * (cos_theta / r1_sq + cos_phi / r2_sq));

        particle.v_x = 0.0;
        particle.v_y = particle.m / DT;

        let q_val = (2.0 * particle.k + 1.0) * base_charge;

        particle.q = match x % 2 {
            0 => q_val,
            _ => -q_val,
        };
        particle.x0 = x_coord;
        particle.y0 = y_coord;
        particle.id = id as f64;
        id += 1;
    }
}

fn initialize_linear(
    n_input: u64,
    grid_size: u64,
    alpha: f64,
    beta: f64,
    horizontal_speed: f64,
    vertical_speed: f64,
    comm: &mpi::topology::SimpleCommunicator,
) -> Vec<Particle> {
    let mut dice = common::RandomDraw::new();
    let mut particles = Vec::<Particle>::new();

    let step = 1.0 / (grid_size as f64);
    let total_weight =
        beta * grid_size as f64 - alpha * 0.5 * step * grid_size as f64 * (grid_size as f64 - 1.0);
    dice.lcg_init();
    for x in 0..grid_size {
        let current_weight = beta - alpha * step * (x as f64);
        for y in 0..grid_size {
            let part_num = dice
                .random_draw(n_input as f64 * (current_weight / total_weight) / grid_size as f64);
            for _p in 0..part_num {
                let mut particle = Particle::default();
                particle.x = x as f64 + REL_X;
                particle.y = y as f64 + REL_Y;
                particle.k = horizontal_speed;
                particle.m = vertical_speed;
                particles.push(particle);
            }
        }
    }
    finalize_distribution(particles.as_mut_slice(), comm);
    particles
}

fn initialize_patch(
    n_input: u64,
    grid_size: u64,
    patch: &BoundingBox,
    horizontal_speed: f64,
    vertical_speed: f64,
    comm: &mpi::topology::SimpleCommunicator,
) -> Vec<Particle> {
    let mut dice = common::RandomDraw::new();

    let total_cells = (patch.right - patch.left + 1) * (patch.top - patch.bottom + 1);
    let particles_per_cell = (n_input as f64) / (total_cells as f64);

    let mut particles = Vec::<Particle>::new();

    dice.lcg_init();

    for x in 0..grid_size {
        for y in 0..grid_size {
            let mut part_num = dice.random_draw(particles_per_cell);
            if x < patch.left || x > patch.right || y < patch.bottom || y > patch.top {
                part_num = 0;
            }
            for _p in 0..part_num {
                let mut particle = Particle::default();
                particle.x = x as f64 + REL_X;
                particle.y = y as f64 + REL_Y;
                particle.k = horizontal_speed;
                particle.m = vertical_speed;
                particles.push(particle);
            }
        }
    }
    finalize_distribution(particles.as_mut_slice(), comm);

    particles
}

fn initialize_geometric(
    n_input: u64,
    grid_size: u64,
    rho: f64,
    horizontal_speed: f64,
    vertical_speed: f64,
    comm: &mpi::topology::SimpleCommunicator,
) -> Vec<Particle> {
    let mut dice = common::RandomDraw::new();

    let mut particles = Vec::<Particle>::new();
    dice.lcg_init();

    let factor =
        n_input as f64 * ((1.0 - rho) / (1.0 - rho.powf(grid_size as f64))) / (grid_size as f64);

    for x in 0..grid_size {
        for y in 0..grid_size {
            let part_num = dice.random_draw(factor * rho.powf(x as f64));
            for _p in 0..part_num {
                let mut particle = Particle::default();
                particle.x = x as f64 + REL_X;
                particle.y = y as f64 + REL_Y;
                particle.k = horizontal_speed;
                particle.m = vertical_speed;
                particles.push(particle);
            }
        }
    }

    finalize_distribution(particles.as_mut_slice(), comm);
    particles
}

fn initialize_sinusoidal(
    n_input: u64,
    grid_size: u64,
    tile: &BoundingBox,
    horizontal_speed: f64,
    vertical_speed: f64,
    comm: &mpi::topology::SimpleCommunicator,
) -> Vec<Particle> {
    let mut dice = common::RandomDraw::new();
    dice.lcg_init();

    let step = PI / (grid_size as f64);
    let mut particles = Vec::<Particle>::new();

    for x in tile.left..tile.right {
        let start_index = tile.bottom + x * grid_size;
        dice.lcg_jump(2 * start_index, 0);
        for y in tile.bottom..tile.top {
            let val = (x as f64 * step).cos();
            let part_num =
                dice.random_draw(2.0 * n_input as f64 * val.powi(2) / grid_size.pow(2) as f64);
            for _p in 0..part_num {
                let mut particle = Particle::default();
                particle.x = x as f64 + REL_X;
                particle.y = y as f64 + REL_Y;
                particle.k = horizontal_speed;
                particle.m = vertical_speed;
                particles.push(particle);
            }
        }
    }

    finalize_distribution(particles.as_mut_slice(), comm);
    particles
}

fn compute_coulomb(x_dist: f64, y_dist: f64, q1: f64, q2: f64) -> (f64, f64) {
    let r2 = x_dist.powi(2) + y_dist.powi(2);
    let r = r2.sqrt();
    let f_coulomb = q1 * q2 / r2;

    let fx = f_coulomb * x_dist / r;
    let fy = f_coulomb * y_dist / r;

    (fx, fy)
}

fn compute_total_force(particle: &mut Particle, grid: &Grid<f64>) -> (f64, f64) {
    let x = particle.x.floor() as usize;
    let y = particle.y.floor() as usize;
    let rel_x = particle.x - particle.x.floor();
    let rel_y = particle.y - particle.y.floor();
    let mut temp_res_x = 0.0;
    let mut temp_res_y = 0.0;

    let (temp_fx, temp_fy) = compute_coulomb(rel_x, rel_y, particle.q, grid.get(x, y));
    temp_res_x += temp_fx;
    temp_res_y += temp_fy;

    let (temp_fx, temp_fy) = compute_coulomb(rel_x, 1.0 - rel_y, particle.q, grid.get(x, y + 1));
    temp_res_x += temp_fx;
    temp_res_y -= temp_fy;

    let (temp_fx, temp_fy) = compute_coulomb(1.0 - rel_x, rel_y, particle.q, grid.get(x + 1, y));
    temp_res_x -= temp_fx;
    temp_res_y += temp_fy;

    let (temp_fx, temp_fy) =
        compute_coulomb(1.0 - rel_x, 1.0 - rel_y, particle.q, grid.get(x + 1, y + 1));
    temp_res_x -= temp_fx;
    temp_res_y -= temp_fy;

    let fx = temp_res_x;
    let fy = temp_res_y;
    (fx, fy)
}

enum Status {
    Failure,
    Success,
}

fn verify_particle(
    particle: &Particle,
    iterations: u64,
    grid: &Grid<f64>,
    grid_size: u64,
) -> Status {
    let disp = (iterations + 1) as f64 * (2.0 * particle.k + 1.0);
    let x_final = match particle.q * grid.get(particle.x0 as usize, particle.y0 as usize) > 0.0 {
        true => particle.x0 + disp,
        false => particle.x0 - disp,
    };
    let y_final = particle.y0 + particle.m * (iterations + 1) as f64;
    let grid_size_f = grid_size as f64;
    let total_it = iterations as f64;

    let x_periodic = (x_final + total_it * (2.0 * particle.k + 1.0) * grid_size_f) % grid_size_f;
    let y_periodic = (y_final + total_it * (particle.m.abs()) * grid_size_f) % grid_size_f;

    if (particle.x - x_periodic).abs() > EPSILON || (particle.y - y_periodic).abs() > EPSILON {
        Status::Failure
    } else {
        Status::Success
    }
}

fn find_owner_simple(
    particle: &Particle,
    width: u64,
    height: u64,
    num_procs_x: u64,
    _i_crit: u64,
    _j_crit: u64,
    _i_leftover: u64,
    _j_leftover: u64,
) -> u64 {
    let x = particle.x.floor() as u64;
    let y = particle.y.floor() as u64;
    let id_x = x / width;
    let id_y = y / height;
    let proc_id = id_y * num_procs_x + id_x;

    proc_id
}
fn find_owner_general(
    particle: &Particle,
    width: u64,
    height: u64,
    num_procs_x: u64,
    i_crit: u64,
    j_crit: u64,
    i_leftover: u64,
    j_leftover: u64,
) -> u64 {
    let x = particle.x.floor() as u64;
    let y = particle.y.floor() as u64;

    let id_x = match x < i_crit {
        true => x / (width + 1),
        false => i_leftover + (x - i_crit) / width,
    };

    let id_y = match y < j_crit {
        true => y / (height + 1),
        false => j_leftover + (y - j_crit) / height,
    };
    let proc_id = id_y * num_procs_x + id_x;
    proc_id
}

// fn add_particle_to_buffer(particle: &Particle, buffer: &Vec<Particle>, position: usize) {

// }

fn main() {
    let args = Args::parse();
    let universe = mpi::initialize().expect("Error initializing MPI");
    let world = universe.world();
    let my_rank = world.rank() as u64;
    let num_procs = world.size() as u64;

    if my_rank == 0 {
        println!("Parallel Research Kernels");
        println!("MPI Particle-in-Cell execution on 2D grid");
        println!("Number of ranks                    = {}", num_procs);
        println!("Load balancing                     = None");
    }

    let grid_size = args.grid_size;
    let grid_patch = BoundingBox {
        left: 0,
        right: grid_size + 1,
        bottom: 0,
        top: grid_size,
    };
    let mut num_procs_y = 0u64;
    let mut num_procs_x = 0u64;
    for procs in (1..((num_procs).isqrt() + 1)).rev() {
        // if my_rank == 0 {
        //     println!("comm_size, procs: {}, {}", comm_size, procs)
        // }
        if num_procs % procs == 0 {
            num_procs_y = num_procs / procs;
            num_procs_x = procs;
            break;
        }
    }
    if my_rank == 0 {
        println!("Grid size                          = {}", args.grid_size);
        println!(
            "Tiles in x/y-direction             = {}/{}",
            num_procs_x, num_procs_y
        );
        println!(
            "Number of particles requested      = {}",
            args.total_particles
        );
        println!("Number of time steps               = {}", args.iterations);
        print!("Initialization mode");
        match args.init_style {
            InitStyle::Sinusoidal => println!("                = SINUSOIDAL"),
            InitStyle::Geometric { attenuation_factor } => {
                println!("            = GEOMETRIC");
                println!("  Attenuation factor           = {:.6}", attenuation_factor)
            }
            InitStyle::Linear {
                negative_slope,
                constant_offset,
            } => {
                println!("            = LINEAR");
                println!("  Negative slope               = {:.6}", negative_slope);
                println!("  Offset                       = {:.6}", constant_offset);
            }
            InitStyle::Patch {
                xleft,
                xright,
                ybottom,
                ytop,
            } => {
                println!("            = PATCH");
                println!(
                    "  Bounding box                 = {}, {}, {}, {}",
                    xleft, xright, ybottom, ytop
                );
            }
        };
        println!(
            "Particle charge semi-increment (k) = {}",
            args.particle_charge_semi_increment
        );
        println!(
            "Vertical velocity              (m) = {}",
            args.vertical_particle_velocity
        );
    }
    let my_rank_x = my_rank % num_procs_x;
    let my_rank_y = my_rank / num_procs_x;

    let k = args.particle_charge_semi_increment as f64;
    let m = args.vertical_particle_velocity as f64;

    let width = grid_size / num_procs_x;
    if (width as f64) < (2.0 * k) {
        if my_rank == 0 {
            panic!(
                "k-value too large: {}, must be no greater than {}",
                k,
                width / 2
            );
        }
    }
    let i_leftover = grid_size % num_procs_x;

    let i_start = match my_rank_x < i_leftover {
        true => (width + 1) * my_rank_x,
        false => (width + 1) * i_leftover + width * (my_rank_x - i_leftover),
    };

    let i_end = match my_rank_x < i_leftover {
        true => i_start + width + 1,
        false => i_start + width,
    };

    let i_crit = (width + 1) * i_leftover;

    let height = grid_size / (num_procs_y as u64);

    if (height as f64) < (2.0 * m) {
        if my_rank == 0 {
            panic!(
                "m-value too large: {}, must not be greater than {}",
                m, height
            );
        }
    }

    let j_leftover = grid_size % num_procs_y;
    let j_start = match my_rank_y < j_leftover {
        true => (height + 1) * my_rank_y,
        false => (height + 1) * j_leftover + height * (my_rank_y - j_leftover),
    };

    let j_end = match my_rank_y < j_leftover {
        true => j_start + height + 1,
        false => j_start + height,
    };

    let j_crit = (height + 1) * j_leftover;

    type FindFunc = fn(&Particle, u64, u64, u64, u64, u64, u64, u64) -> u64;

    let find_owner: FindFunc = match i_crit == 0 && j_crit == 0 {
        true => {
            if my_rank == 0 {
                println!("Rank search mode used              = simple");
            }
            find_owner_simple
        }
        false => {
            if my_rank == 0 {
                println!("Rank search mode used              = general");
            }
            find_owner_general
        }
    };

    let my_tile = BoundingBox {
        left: i_start,
        right: i_end,
        bottom: j_start,
        top: j_end,
    };

    let mut nbr = [0u64; 8];

    nbr[0] = match my_rank_x == 0 {
        true => my_rank + num_procs_x - 1,
        false => my_rank - 1,
    };
    nbr[1] = match my_rank_x == num_procs_x - 1 {
        true => my_rank - num_procs_x + 1,
        false => my_rank + 1,
    };
    nbr[2] = match my_rank_y == num_procs_y - 1 {
        true => my_rank + num_procs_x - num_procs,
        false => my_rank + num_procs_x,
    };
    nbr[3] = match my_rank_y == 0 {
        true => my_rank - num_procs_x + num_procs,
        false => my_rank - num_procs_x,
    };
    nbr[4] = match my_rank_y == num_procs_y - 1 {
        true => nbr[0] + num_procs_x - num_procs,
        false => nbr[0] + num_procs_x,
    };
    nbr[5] = match my_rank_y == num_procs_y - 1 {
        true => nbr[1] + num_procs_x - num_procs,
        false => nbr[1] + num_procs_x,
    };
    nbr[6] = match my_rank_y == 0 {
        true => nbr[0] - num_procs_x + num_procs,
        false => nbr[0] - num_procs_x,
    };
    nbr[7] = match my_rank_y == 0 {
        true => nbr[1] - num_procs_x + num_procs,
        false => nbr[1] - num_procs_x,
    };
    let grid = initialize_grid(&my_tile);

    let mut particles = match args.init_style {
        InitStyle::Geometric { attenuation_factor } => initialize_geometric(
            args.total_particles,
            grid_size,
            attenuation_factor,
            k,
            m,
            &world,
        ),
        InitStyle::Sinusoidal => {
            initialize_sinusoidal(args.total_particles, grid_size, &my_tile, k, m, &world)
        }
        InitStyle::Linear {
            negative_slope,
            constant_offset,
        } => {
            if constant_offset < 0.0 || constant_offset < negative_slope {
                panic!("ERROR: linear profile gives negative density");
            }
            initialize_linear(
                args.total_particles,
                grid_size,
                negative_slope,
                constant_offset,
                k,
                m,
                &world,
            )
        }
        InitStyle::Patch {
            xleft,
            xright,
            ybottom,
            ytop,
        } => {
            let patch = BoundingBox {
                left: xleft,
                right: xright,
                bottom: ybottom,
                top: ytop,
            };
            if bad_patch(&patch, &grid_patch) {
                panic!("ERROR: inconsistent initial patch");
            };
            initialize_patch(args.total_particles, grid_size, &patch, k, m, &world)
        }
    };
    if my_rank == 0 {
        let mut total_parts = 0;
        world.process_at_rank(ROOT_RANK).reduce_into_root(
            &particles.len(),
            &mut total_parts,
            SystemOperation::sum(),
        );
        println!("Number of particles placed         = {}", total_parts);
    } else {
        world
            .process_at_rank(ROOT_RANK)
            .reduce_into(&particles.len(), SystemOperation::sum());
    }
    // let timer = Instant::now();
    // let mut t0 = timer.elapsed();

    // let mut sendbuf = Vec::<Particle>::with_capacity(10);

    // let mut sendbuf = [Vec::<Particle>::with_capacity(10); 8];

    let mut sendbuf: [Vec<Particle>; 8] =
        core::array::from_fn(|_| Vec::<Particle>::with_capacity(10));
    let mut recvbuf: [Vec<Particle>; 8] =
        core::array::from_fn(|_| Vec::<Particle>::with_capacity(10));

    for it in 0..args.iterations + 1 {
        //     if it == 1 {
        //         t0 = timer.elapsed();
        //     }
        for particle in particles.iter_mut() {
            let (fx, fy) = compute_total_force(particle, &grid);
            let ax = fx * MASS_INV;
            let ay = fy * MASS_INV;
            let x_disp = particle.x + particle.v_x * DT + 0.5 * ax * DT.powi(2) + grid_size as f64;
            let y_disp = particle.y + particle.v_y * DT + 0.5 * ay * DT.powi(2) + grid_size as f64;
            particle.x = x_disp % grid_size as f64;
            particle.y = y_disp % grid_size as f64;

            particle.v_x += ax * DT;
            particle.v_y += ay * DT;
            let owner = find_owner(
                &particle, width, height, num_procs, i_crit, j_crit, i_leftover, j_leftover,
            );
            if owner == nbr[0] {
                sendbuf[0].push(*particle);
            } else if owner == nbr[1] {
                sendbuf[1].push(*particle);
            } else if owner == nbr[2] {
                sendbuf[2].push(*particle);
            } else if owner == nbr[3] {
                sendbuf[3].push(*particle);
            } else if owner == nbr[4] {
                sendbuf[4].push(*particle);
            } else if owner == nbr[5] {
                sendbuf[5].push(*particle);
            } else if owner == nbr[6] {
                sendbuf[6].push(*particle);
            } else if owner == nbr[7] {
                sendbuf[7].push(*particle);
            } else if owner == nbr[8] {
                sendbuf[8].push(*particle);
            } else {
                panic!(
                    "Could not find neighbor owner of particle in tile {}",
                    owner
                );
            }
            mpi::request::multiple_scope(
                16,
                |scope, coll: &mut RequestCollection<'_, Vec<Particle>>| {
                    for i in 0..8 {
                        let sreq = world
                            .process_at_rank(nbr[i] as i32)
                            .immediate_send(scope, &sendbuf[i]);
                        coll.add(sreq);
                        let rreq = world.this_process().immediate_receive_into(scope, buf);
                    }
                },
            );
        }
    }
    // let t1 = timer.elapsed();
    // let dt = (t1.checked_sub(t0)).unwrap();
    // let pic_time = dt.as_secs_f64();

    // let mut result = true;
    // for particle in particles.iter() {
    //     match verify_particle(particle, args.iterations, &grid, grid_size) {
    //         Status::Failure => {
    //             result = false;
    //             break;
    //         }
    //         _ => (),
    //     };
    // }

    // match result {
    //     true => {
    //         let average_time = (args.iterations * args.total_particles) as f64 / pic_time;
    //         println!("Solution validates");
    //         println!("Rate (Mparticles_moved/s): {:.6}", 1e-6 * average_time);
    //     }
    //     false => println!("Solution does not validate"),
    // };
}
