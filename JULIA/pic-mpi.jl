using Polyester
using MPI

MPI.Init()
comm = MPI.COMM_WORLD

const global MASS_INV::Float64 = 1.0
const global Q::Float64 = 1.0
const global epsilon::Float64 = 0.00001
const global DT::Float64 = 1.0

const global REL_X::Float64 = 0.5
const global REL_Y::Float64 = 0.5
const global random_seed::UInt64 = 27182818285
const global LCG_a::UInt64 = 6364136223846793005
const global LCG_c::UInt64 = 1442695040888963407
const global ROOT = 0

@enum InitType GEOMETRIC SINUSOIDAL LINEAR PATCH UNDEFINED

function parse_init_type(init_type_str::String)
  if init_type_str == "GEOMETRIC"
    return GEOMETRIC
  elseif init_type_str == "SINUSOIDAL"
    return SINUSOIDAL
  elseif init_type_str == "LINEAR"
    return LINEAR
  elseif init_type_str == "PATCH"
    return PATCH
  else
    return UNDEFINED
  end
end

mutable struct Particle
  x::Float64
  y::Float64
  v_x::Float64
  v_y::Float64
  q::Float64

  x0::Float64
  y0::Float64
  k::Int64
  m::Int64
end

sizes = [sizeof(t) for t in Particle.types]
displacements = [8 * i for i = 1:length(sizes)]
types = [MPI.Datatype(t) for t in Particle.types]

ParticleMPI = MPI.Types.create_struct(sizes, displacements, types)
MPI.Types.commit!(ParticleMPI)

struct BoundingBox
  left::UInt64
  right::UInt64
  bottom::UInt64
  top::UInt64
end

# Pretty print BoundingBox struct
Base.show(io::IO, bb::BoundingBox) = print(io, "$(bb.left), $(bb.right), $(bb.bottom), $(bb.top)")

function bad_patch(patch::BoundingBox, patch_contain::BoundingBox)
  if patch.left >= patch.right || patch.bottom >= patch.top
    return true
  end

  if patch.left < patch_contain.left || patch.right > patch_contain.right
    return true
  end

  if patch.bottom < patch_contain.bottom || patch.top > patch_contain.top
    return true
  end
  return false
end

function initialize_grid(length)
  q_grid = zeros(Float64, length + 1, length + 1)
  for i in 1:(length+1)
    for j in 1:(length+1)
      q_grid[i, j] = (i - 1) % 2 == 1 ? -Q : Q
    end
  end
  return q_grid
end

function finish_distribution(n_placed, particles)
  for pi = 1:n_placed
    x_coord = particles[pi].x
    y_coord = particles[pi].y
    rel_x = x_coord % 1.0
    rel_y = y_coord % 1.0
    x = UInt64(floor(x_coord))
    r1_sq = rel_y * rel_y + rel_x * rel_x
    r2_sq = rel_y * rel_y + (1.0 - rel_x) * (1.0 - rel_x)
    cos_theta = rel_x / sqrt(r1_sq)
    cos_phi = (1.0 - rel_x) / sqrt(r2_sq)
    base_charge = 1.0 / ((DT * DT) * Q * (cos_theta / r1_sq + cos_phi / r2_sq))

    particles[pi].v_x = 0.0
    particles[pi].v_y = particles[pi].m / DT
    particles[pi].q = (x % 2 == 0) ? (2 * particles[pi].k + 1) * base_charge : -1.0 * (2 * particles[pi].k + 1) * base_charge
    particles[pi].x0 = x_coord
    particles[pi].y0 = y_coord
  end
end

function lcg_next(bound, seed)
  new_seed = LCG_a * seed + LCG_c
  return new_seed % bound, new_seed
end

function random_draw(mu, seed)
  two_pi = 2.0 * 3.14159265358979323846
  rand_max = typemax(UInt64)
  rand_div = 1 / rand_max
  denominator = typemax(UInt32)
  if mu >= 1.0
    sigma = mu * 0.15
    val0, seed = lcg_next(rand_max, seed)
    val1, seed = lcg_next(rand_max, seed)
    u0 = val0 * rand_div
    u1 = val1 * rand_div

    z0 = sqrt(-2.0 * log(u0)) * cos(two_pi * u1)
    _ = sqrt(-2.0 * log(u0)) * sin(two_pi * u1)
    return UInt64(floor(z0 * sigma + mu + 0.5)), seed
  else
    numerator = UInt32(floor(mu * denominator))
    _, seed = lcg_next(denominator, seed)
    i1, seed = lcg_next(denominator, seed)

    return i1 <= numerator ? 1 : 0, seed
  end
end

function initialize_geometric(n_input, L, rho, k, m)
  A = n_input * ((1.0 - rho) / (1.0 - (rho^L))) / L
  n_placed = 0
  seed = random_seed
  for x = 0:(L-1)
    for _ = 0:(L-1)
      new_amount, seed = random_draw(A * rho^x, seed)
      n_placed += new_amount
    end
  end
  particles = [Particle(0, 0, 0, 0, 0, 0, 0, 0, 0) for i = 1:n_placed]
  seed = random_seed
  pi = 1
  for x = 0:(L-1)
    for y = 0:(L-1)
      actual_particles, seed = random_draw(A * rho^x, seed)
      for _ = 1:actual_particles
        particles[pi].x = x + REL_X
        particles[pi].y = y + REL_Y
        particles[pi].k = k
        particles[pi].m = m
        pi += 1
      end
    end
  end
  finish_distribution(n_placed, particles)
  return particles, n_placed
end

@inline function compute_coulomb(x_dist, y_dist, q1, q2)
  r2 = x_dist * x_dist + y_dist * y_dist
  r = sqrt(r2)
  f_coulomb = q1 * q2 / r2
  fx = f_coulomb * x_dist / r
  fy = f_coulomb * y_dist / r

  return fx, fy

end

@inline function compute_total_force(particle, q_grid)
  tmp_res_x = 0.0
  tmp_res_y = 0.0

  x = UInt64(floor(particle.x))
  y = UInt64(floor(particle.y))
  y_idx = y + 1
  x_idx = x + 1

  rel_x = particle.x - x
  rel_y = particle.y - y

  tmp_fx, tmp_fy = compute_coulomb(rel_x, rel_y, particle.q, q_grid[x_idx, y_idx])
  tmp_res_x += tmp_fx
  tmp_res_y += tmp_fy

  tmp_fx, tmp_fy = compute_coulomb(rel_x, 1.0 - rel_y, particle.q, q_grid[x_idx, y_idx+1])
  tmp_res_x += tmp_fx
  tmp_res_y -= tmp_fy

  tmp_fx, tmp_fy = compute_coulomb(1.0 - rel_x, rel_y, particle.q, q_grid[x_idx+1, y_idx])
  tmp_res_x -= tmp_fx
  tmp_res_y += tmp_fy

  tmp_fx, tmp_fy = compute_coulomb(1.0 - rel_x, 1.0 - rel_y, particle.q, q_grid[x_idx+1, y_idx+1])
  tmp_res_x -= tmp_fx
  tmp_res_y -= tmp_fy

  fx = tmp_res_x
  fy = tmp_res_y

  return fx, fy
end

function verify_particle(particle, iterations, q_grid, grid_dimension)
  y = UInt64(floor(particle.y0)) + 1
  x = UInt64(floor(particle.x0)) + 1

  disp = (iterations + 1) * (2 * particle.k + 1)
  x_final = ((particle.q * q_grid[x, y]) > 0) ? particle.x0 + disp : particle.x0 - disp
  y_final = particle.y0 + particle.m * (iterations + 1)

  x_periodic = (x_final + (iterations + 1) * (2 * particle.k + 1) * grid_dimension) % grid_dimension
  y_periodic = (y_final + (iterations + 1) * abs(particle.m) * grid_dimension) % grid_dimension

  if (abs(particle.x - x_periodic) > epsilon) || (abs(particle.y - y_periodic) > epsilon)
    return false
  end
  return true

end

function get_coords(rank, comm_size)
  return div(rank, comm_size), rem(rank, comm_size)
end

function main(arg_vector::Vector{String})
  my_rank = MPI.Comm_rank(comm)
  comm_size = MPI.Comm_size(comm)

  if my_rank == ROOT
    println("MPI Julia Particle-in-Cell execution on 2D grid")
    if length(arg_vector) < 6
      println("Usage: julia $(PROGRAM_FILE) <#simulation steps> <grid size> <#particles> <k (particle charge semi-increment)> <m (vertical particle velocity)> <init mode> <init parameters>")
      println("init mode \"GEOMETRIC\"  parameters: <attenuation factor>")
      println("          \"SINUSOIDAL\" parameters: none")
      println("          \"LINEAR\"     parameters: <negative slope> <constant offset>")
      println("          \"PATCH\"      parameters: <xleft> <xright> <ybottom> <ytop>")
      exit(0)
    end
  end

  iterations = parse(UInt64, arg_vector[1])
  if iterations < 1
    println("ERROR: Number of iterations must be positive: $(iterations)")
    exit(-1)
  end

  grid_dimensions = parse(UInt64, arg_vector[2])
  if grid_dimensions < 1 || grid_dimensions % 2 == 1
    println("ERROR: Number of grid cells must be positive and even: $(grid_dimensions)")
    exit(-1)
  end

  grid_patch = BoundingBox(0, grid_dimensions + 1, 0, grid_dimensions + 1)
  number_of_particles = parse(UInt64, arg_vector[3])
  if number_of_particles < 1
    println("ERROR: Number of particles must be positive: $(number_of_particles)")
    exit(-1)
  end

  k = parse(UInt64, arg_vector[4])
  m = parse(Int64, arg_vector[5])

  particle_mode = parse_init_type(arg_vector[6])
  if particle_mode == GEOMETRIC
    if length(arg_vector) < 7
      println("ERROR: Not enough arguments for GEOMETRIC")
      exit(-1)
    end
    rho = parse(Float64, arg_vector[7])
  elseif particle_mode == LINEAR
    if length(arg_vector) < 8
      println("ERROR: Not enough arguments for LINEAR initialization")
      exit(-1)
    end
    alpha = parse(Float64, arg_vector[7])
    beta = parse(Float64, arg_vector[8])
    if beta < 0 || beta < alpha
      println("ERROR: linear profile gives negative particle density")
      exit(-1)
    end
  elseif particle_mode == PATCH
    if length(arg_vector) < 10
      println("ERROR: Not enough arguments for PATCH initialization")
      exit(-1)
    end
    patch_dims = map(x -> parse(UInt64, x), arg_vector[7:10])
    init_patch = BoundingBox(patch_dims...)
    if bad_patch(init_patch, grid_patch)
      println("ERROR: inconsistent initial patch")
      exit(-1)
    end
  end

  if my_rank == ROOT
    println("Number of ranks                = $(comm_size)")
    println("Grid size                      = $(grid_dimensions)")
    println("Number of particles requested  = $(number_of_particles)")
    println("Number of time steps           = $(iterations)")
    println("Initialization mode            = $(particle_mode)")

    if particle_mode == GEOMETRIC
      println("  Attenuation factor           = $(rho)")
    elseif particle_mode == LINEAR
      println("Negative slope                 = $(alpha)")
      println("Offset                         = $(beta)")
    elseif particle_mode == PATCH
      println("Bounding box                   = $(init_patch)")
    elseif particle_mode == UNDEFINED
      println("ERROR: Unsupported particle initializating mode")
      exit(-1)
    end
    println("Particle charge semi-increment = $(k)")
    println("Vertical velocity              = $(m)")
  end
  q_grid = initialize_grid(grid_dimensions)

  num_procsx = isqrt(comm_size + 1)
  num_procsy = 0
  while num_procsx > 0
    if comm_size % num_procsx == 0
      num_procsy = div(comm_size, num_procsx)
      break
    end
    num_procsx -= 1
  end

  if my_rank == ROOT
    println("$(num_procsx), $(num_procsy)")
    width = div(grid_dimensions, num_procsx)
    # if particle_mode == GEOMETRIC
    #   particles, n_placed = initialize_geometric(number_of_particles, grid_dimensions, rho, k, m)
    # end
    #
    # println("Number of particles placed     = $(n_placed)")
    # precompile(compute_coulomb, (Float64, Float64, Float64, Float64))
    # precompile(compute_total_force, (Particle, Matrix{Float64}))
    # mul_const = DT * DT * 0.5
    #
    # pic_time = @elapsed for i = 0:iterations
    #   @batch for pi = 1:n_placed
    #     @inbounds fx, fy = compute_total_force(particles[pi], q_grid)
    #     ax = fx * MASS_INV
    #     ay = fy * MASS_INV
    #     @inbounds particles[pi].x = (particles[pi].x + particles[pi].v_x * DT + ax * mul_const + grid_dimensions) % grid_dimensions
    #     @inbounds particles[pi].y = (particles[pi].y + particles[pi].v_y * DT + ay * mul_const + grid_dimensions) % grid_dimensions
    #     @inbounds particles[pi].v_x += ax * DT
    #     @inbounds particles[pi].v_y += ay * DT
    #   end
    # end
    #
    # correctness = all(part -> verify_particle(part, iterations, q_grid, grid_dimensions), particles)
    #
    # if correctness == 1
    #   println("Solution validates")
    #   println("time: $(pic_time)")
    #   avg_time = n_placed * iterations / pic_time
    #   println("Rate (Mparticles_moved/s): $(1.0e-6 * avg_time)")
    # else
    #   println("Solution does not validate")
    # end
  end
end

main(ARGS)
