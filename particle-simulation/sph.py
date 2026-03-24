import taichi as ti
import math
import time

ti.init(
    arch=ti.vulkan,
    default_ip=ti.i32,
    offline_cache=True,
    verbose=False
)

# 参数定义
N_PARTICLES = 120000
N_ARMS = 4
GALAXY_RADIUS = 14.0
VIEW_RADIUS = 17.5
PARTICLE_RADIUS = 0.8
RENDER_RATIO = 0.65
RENDER_COUNT = int(N_PARTICLES * RENDER_RATIO)
RENDER_INTERVAL = 1

CORE_RATIO = 0.4
HALO_RATIO = 0.10
ARM_SPREAD = 0.42
SPIRAL_TIGHTNESS = 3
BASE_OMEGA = 1
ROTATION_SIGN = -1.0

# 粒子属性场
pos = ti.Vector.field(2, dtype=ti.f32, shape=N_PARTICLES)
radius = ti.field(dtype=ti.f32, shape=N_PARTICLES)
base_angle = ti.field(dtype=ti.f32, shape=N_PARTICLES)
omega = ti.field(dtype=ti.f32, shape=N_PARTICLES)
phase = ti.field(dtype=ti.f32, shape=N_PARTICLES)
kind = ti.field(dtype=ti.i32, shape=N_PARTICLES)  # 0=core, 1=disk, 2=halo
sim_time = ti.field(dtype=ti.f32, shape=())
render_pos = ti.Vector.field(3, dtype=ti.f32, shape=RENDER_COUNT)
render_color = ti.Vector.field(3, dtype=ti.f32, shape=RENDER_COUNT)


@ti.kernel
def init_particles():
    """初始化为核球 + 旋臂盘 + 外晕三层结构。"""
    for i in range(N_PARTICLES):
        u = ti.random()
        p = ti.random()
        r = 0.0
        theta0 = 0.0
        k = ti.cast(0, ti.i32)

        if u < CORE_RATIO:
            # 中心核球：更密集、角度更随机
            r = GALAXY_RADIUS * 0.22 * ti.sqrt(p)
            theta0 = 2.0 * math.pi * ti.random()
            k = 0
        elif u < CORE_RATIO + HALO_RATIO:
            # 外晕：分布稀疏、较暗
            r = GALAXY_RADIUS * (0.76 + 0.24 * p)
            theta0 = 2.0 * math.pi * ti.random()
            k = 2
        else:
            # 盘面旋臂：半径偏向外层，按臂号组织
            arm = ti.cast(ti.floor(ti.random() * N_ARMS), ti.i32)
            r = GALAXY_RADIUS * ti.pow(p, 0.65)
            arm_base = 2.0 * math.pi * ti.cast(arm, ti.f32) / ti.cast(N_ARMS, ti.f32)
            winding = SPIRAL_TIGHTNESS * ti.log(1.0 + r)
            jitter = (ti.random() - 0.5) * ARM_SPREAD * (0.35 + r / GALAXY_RADIUS)
            theta0 = arm_base + winding + jitter
            k = 1

        radius[i] = r
        base_angle[i] = theta0
        omega[i] = BASE_OMEGA / (0.35 + ti.sqrt(ti.max(r, 1e-3)))
        phase[i] = 2.0 * math.pi * ti.random()
        kind[i] = k


@ti.kernel
def update_particles():
    """按差速旋转更新位置，保持旋臂形态稳定。"""
    for i in range(N_PARTICLES):
        r = radius[i]
        t = sim_time[None]
        theta = base_angle[i] + ROTATION_SIGN * omega[i] * t

        # 加一点细微扰动，避免机械感
        ripple = 0.035 * r * ti.sin(1.8 * t + phase[i])
        rr = ti.max(0.0, r + ripple)

        x = rr * ti.cos(theta)
        y = rr * ti.sin(theta)

        # 中央棒状势的轻微视觉效果
        bar = 0.18 * ti.exp(-r * 0.45) * ti.sin(2.0 * theta + 0.15 * t)
        x += 0.42 * bar
        y -= 0.24 * bar

        pos[i] = ti.Vector([x, y])


@ti.func
def clamp01(x):
    return ti.min(1.0, ti.max(0.0, x))


@ti.kernel
def build_render_buffer():
    """GPU 侧构建粒子位置和颜色，用于 GGUI 直接渲染。"""
    t = sim_time[None]
    for j in range(RENDER_COUNT):
        i = (j * N_PARTICLES) // RENDER_COUNT
        p = pos[i]
        r = radius[i]
        k = kind[i]
        ph = phase[i]

        render_pos[j] = ti.Vector([p.x, p.y, 0.0])

        theta = ti.atan2(p.y, p.x)
        radial = clamp01(1.0 - ti.pow(r / GALAXY_RADIUS, 0.70))
        radial = ti.max(radial, 0.05)
        twinkle = 0.84 + 0.16 * ti.sin(2.3 * t + ph * 2.0)

        arm_phase = ti.cast(N_ARMS, ti.f32) * theta - SPIRAL_TIGHTNESS * ti.log(1.0 + ti.max(r, 1e-4))
        arm_glow = clamp01(0.5 + 0.5 * ti.sin(arm_phase - 1.9 * t))
        lum = clamp01(radial * twinkle * (0.72 + 0.62 * arm_glow))

        red_c = 0.0
        green_c = 0.0
        blue_c = 0.0
        if k == 0:
            red_c = 1.00
            green_c = 0.92
            blue_c = 0.84
        elif k == 1:
            red_c = 0.26 + 0.18 * arm_glow
            green_c = 0.86 + 0.12 * arm_glow
            blue_c = 1.00
        else:
            red_c = 0.18
            green_c = 0.42
            blue_c = 0.86

        red = ti.cast(clamp01(lum * red_c) * 255.0, ti.u32)
        green = ti.cast(clamp01(lum * green_c) * 255.0, ti.u32)
        blue = ti.cast(clamp01(lum * blue_c) * 255.0, ti.u32)
        render_color[j] = ti.Vector([
            ti.cast(red, ti.f32) / 255.0,
            ti.cast(green, ti.f32) / 255.0,
            ti.cast(blue, ti.f32) / 255.0,
        ])


def render_particles(scene: ti.ui.Scene):
    scene.particles(render_pos, radius=PARTICLE_RADIUS * 0.12, per_vertex_color=render_color)

# 主程序入口
if __name__ == "__main__":
    print("🌀 正在初始化星系粒子...")
    init_particles()

    window = ti.ui.Window("Galaxy", (900, 900), vsync=False)
    canvas = window.get_canvas()
    camera = ti.ui.Camera()
    scene = window.get_scene()
    camera.position(0.0, 0.0, 36.0)
    camera.lookat(0.0, 0.0, 0.0)

    print("🔄 开始模拟演化...")
    frame = 0
    fps_ema = 0.0
    prev_t = time.perf_counter()
    while window.running:
        t = frame * 0.018
        sim_time[None] = t
        update_particles()

        if frame % RENDER_INTERVAL == 0 or frame == 0:
            build_render_buffer()

        scene.set_camera(camera)
        scene.ambient_light((0.32, 0.42, 0.58))
        scene.point_light(pos=(0.0, 0.0, 24.0), color=(1.0, 1.0, 1.0))
        render_particles(scene)
        canvas.set_background_color((0.012, 0.027, 0.065))
        canvas.scene(scene)

        now = time.perf_counter()
        inst_fps = 1.0 / max(1e-6, now - prev_t)
        prev_t = now
        if fps_ema <= 0.0:
            fps_ema = inst_fps
        else:
            fps_ema = fps_ema * 0.92 + inst_fps * 0.08
        window.show()
        frame += 1

    print("✨ 模拟完成！请查看可视化结果")

