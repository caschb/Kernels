const global Q = 1.0
const global epsilon = 0.00001
const global DT = 1.0

const global REL_X = 0.5
const global REL_Y = 0.5

@enum InitType GEOMETRIC SINUSOIDAL LINEAR PATCH UNDEFINED

struct Point
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

struct BoundingBox
  left::UInt64
  right::UInt64
  bottom::UInt64
  top::UInt64
end

function initialize_grid(length)
  q_grid = zeros(Float64, length, length)
  for i in 1:length
    for j in 1:length
        q_grid[i, j] = i % 2 == 1 ? -Q : Q
    end
  end
  return q_grid
end

function main()
  println("Julia Particle-in-Cell execution on 2D grid")
  iterations = 100
  grid_dimensions = 10
  init_mode = GEOMETRIC

  grid_patch = BoundingBox(0, grid_dimensions+1, 0, grid_dimensions+1)
end

main()
