import taichi as ti
ti.init(arch=ti.gpu)

n = 500000
dt = 0.002

cell_size = 0.02
grid_res = 64
grid = ti.field(ti.i32, shape=(grid_res, grid_res))
grid_count = ti.field(ti.i32, shape=(grid_res, grid_res))
grid_particles = ti.field(ti.i32, shape=(grid_res, grid_res, 128))


pos = ti.Vector.field(2, ti.f32, shape=n)
vel = ti.Vector.field(2, ti.f32, shape=n)
active = ti.field(ti.i32, shape=n)
colors = ti.Vector.field(3, ti.f32, shape=n)

spawn_x = 0.05
spawn_y = 0.5

obstacle_center = ti.Vector([0.5, 0.5])
obstacle_radius = 0.12

spawn_ptr = ti.field(ti.i32, shape=())

@ti.kernel
def build_grid():
    for i, j in grid:
        grid_count[i, j] = 0

    for p in range(n):
        if active[p] == 1:
            cell = ti.Vector([int(pos[p].x / cell_size),
                              int(pos[p].y / cell_size)])
            if 0 <= cell.x < grid_res and 0 <= cell.y < grid_res:
                idx = ti.atomic_add(grid_count[cell.x, cell.y], 1)
                if idx < 128:
                    grid_particles[cell.x, cell.y, idx] = p


@ti.kernel
def init():
    for i in range(n):
        active[i] = 0

@ti.kernel
def spawn_particles(num: int):
    for _ in range(num):  # 每帧发射 num 个粒子
        idx = ti.atomic_add(spawn_ptr[None], 1) % n
        if active[idx] == 0:
            pos[idx] = ti.Vector([spawn_x, spawn_y + (ti.random()-0.5)*0.2])
            vel[idx] = ti.Vector([1.5 + ti.random()*0.5, (ti.random()-0.5)*0.2])
            active[idx] = 1


@ti.kernel
def update():
    for i in range(n):
        if active[i] == 0:
            continue

        h = 0.03
        pressure_strength = 40.0
        viscosity_strength = 8.0

        cell = ti.Vector([int(pos[i].x / cell_size),
                        int(pos[i].y / cell_size)])

        for ox in ti.static(range(-1, 2)):
            for oy in ti.static(range(-1, 2)):
                cx = cell.x + ox
                cy = cell.y + oy
                if 0 <= cx < grid_res and 0 <= cy < grid_res:
                    count = grid_count[cx, cy]
                    for k in range(count):
                        j = grid_particles[cx, cy, k]
                        if j != i and active[j] == 1:
                            rij = pos[i] - pos[j]
                            r = rij.norm()

                            if r < h:
                                # 压力力（避免重叠）
                                vel[i] += rij.normalized() * (h - r) * pressure_strength * dt

                                # 黏性扩散（速度传递）
                                vel[i] += (vel[j] - vel[i]) * viscosity_strength * dt
        # 简化流体动力：黏性 + 随机扰动
        vel[i] *= 0.997
        vel[i] += ti.Vector([1, 0.0]) * dt  # 水平驱动

        # 位置更新
        pos[i] += vel[i] * dt

        # 障碍物碰撞（圆形）
        d = pos[i] - obstacle_center
        if d.norm() < obstacle_radius:
            nrm = d.normalized()
            pos[i] = obstacle_center + nrm * obstacle_radius
            vel[i] = vel[i] - 2 * vel[i].dot(nrm) * nrm
            vel[i] *= 0.6

        # 出界则回收
        if pos[i].x > 1.2 or pos[i].y < -0.2 or pos[i].y > 1.2:
            active[i] = 0

        # 速度着色（蓝→红）
        speed = vel[i].norm()
        t = min(speed * 1, 1.0)

        r = 0.2*(1-t) + 1.0*t
        g = 0.6*(1-t) + 0.2*t
        b = 1.0*(1-t) + 0.1*t
        colors[i] = ti.Vector([r, g, b])

window = ti.ui.Window("Turbulence Simulation", (1600, 800))
canvas = window.get_canvas()

init()

while window.running:
    spawn_particles(80)   # 每帧发射 80 个粒子
    build_grid() 
    update()

    canvas.set_background_color((0.02, 0.03, 0.06))
    canvas.circles(pos, radius=0.002, per_vertex_color=colors)
    window.show()
