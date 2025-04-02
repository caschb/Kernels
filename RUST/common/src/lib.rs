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
use std::f64::consts::PI;

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;
const LCG_SEED: u64 = 27182818285;
const NMAX: usize = 64;

#[derive(Debug)]
pub struct RandomDraw {
    pub lcg_seed: u64,
    pub lcg_a: [u64; NMAX],
}

impl RandomDraw {
    pub fn new() -> Self {
        RandomDraw {
            lcg_seed: LCG_SEED,
            lcg_a: [0u64; NMAX],
        }
    }

    pub fn lcg_init(&mut self) {
        self.lcg_seed = LCG_SEED;
        self.lcg_a[0] = LCG_A;
        for i in 1..self.lcg_a.len() {
            self.lcg_a[i] = self.lcg_a[i - 1].pow(2);
        }
    }

    pub fn lcg_next(&mut self, bound: u64) -> u64 {
        let a = LCG_A;
        let c = LCG_C;
        self.lcg_seed = self.lcg_seed.wrapping_mul(a).wrapping_add(c);
        self.lcg_seed % bound
    }

    fn tail(&self, mut x: u64) -> u64 {
        if x == 0 {
            return x;
        }
        let mut result = 1u64;
        let x2 = x;
        while x > 1 {
            x >>= 1;
            result <<= 1;
        }
        return x2 - result;
    }

    fn sum_power(&self, k: i64) -> u64 {
        if k == 0 {
            return LCG_A;
        } else {
            return self.sum_power(k - 1) * (1 + self.lcg_a[(k - 1) as usize]);
        }
    }

    fn log(&self, mut n: u64) -> i64 {
        let mut result = 0i64;
        while n > 1 {
            n >>= 1;
            result += 1;
        }
        result
    }

    fn sumk(&self, n: u64) -> u64 {
        if n == 0 {
            return 0u64;
        }
        let head = self.sum_power(self.log(n));
        let tail_n = self.tail(n);
        if tail_n == 0 {
            return head;
        }
        head + self.lcg_a[self.log(n) as usize] * self.sumk(tail_n)
    }

    pub fn lcg_jump(&mut self, m: u64, bound: u64) {
        let mut lcg_power = [0u64; NMAX];
        self.lcg_seed = LCG_SEED;

        match m {
            0 => return,
            1 => {
                self.lcg_next(bound);
                return;
            }
            _ => (),
        };
        let mut mm = m;
        let mut index = 0;

        while mm > 0 {
            lcg_power[index] = mm & 1;
            index += 1;
            mm >>= 1;
        }
        let mut s_part = 1u64;
        for i in 0..index {
            if lcg_power[i] != 0 {
                s_part *= self.lcg_a[i];
            }
        }
        self.lcg_seed = s_part * self.lcg_seed + (self.sumk(m - 1) + 1) * LCG_C;
    }

    pub fn random_draw(&mut self, mu: f64) -> u64 {
        let rand_max = u64::MAX;
        let rand_div = 1.0 / rand_max as f64;
        let denominator = u32::MAX as u64;
        let two_pi = 2.0 * PI;

        if mu >= 1.0 {
            let sigma = mu * 0.15;
            let u0 = self.lcg_next(rand_max) as f64 * rand_div;
            let u1 = self.lcg_next(rand_max) as f64 * rand_div;
            let z0 = (-2.0 * u0.ln()).sqrt() * (two_pi * u1).cos();
            return (z0 * sigma + mu + 0.5) as u64;
        } else {
            let numerator = (mu * denominator as f64) as u64;
            self.lcg_next(denominator); // Called but result ignored
            let i1 = self.lcg_next(denominator);
            return (i1 <= numerator) as u64;
        }
    }
}
