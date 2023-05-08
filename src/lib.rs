use camera::Camera;
use hit::Hit;
use itertools::iproduct;
use material::Material;
use ndarray::Array3;
use numpy::ToPyArray;
use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods, pymodule,
    types::{PyModule, PyTuple},
    PyResult, Python,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ray::Ray;
use rayon::prelude::*;
use vec3::{Color, Point3, Vec3};
use world::{Object, ObjectKind, World};

mod camera;
mod hit;
mod material;
mod ray;
mod vec3;
mod world;

pub fn random_scene(rng: &mut ChaCha8Rng) -> World {
    let mut scene = Vec::new();

    let ground = Object {
        kind: ObjectKind::Plane {
            origin: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, -1.0, 0.0),
        },
        material: Material::Lambertian {
            albedo: Color::new(0.5, 0.5, 0.5),
        },
    };
    scene.push(ground);

    let light = Object {
        kind: ObjectKind::Sphere {
            origin: Point3::new(2.5, 10.0, 0.0),
            radius: 2.5,
        },
        material: Material::Diffuse {
            albedo: Color::new(1.0, 1.0, 1.0) * 10.0,
        },
    };
    scene.push(light);

    for a in -2..=2 {
        for b in -2..=2 {
            let origin = Point3::new(
                (a as f64) + rng.gen_range(0.0..0.9),
                0.5,
                (b as f64) + rng.gen_range(0.0..0.9),
            );

            let radius: f64 = rng.gen_range(0.4..0.5);

            let kind = ObjectKind::Sphere { origin, radius };

            let material = Material::Lambertian {
                albedo: Color::random(rng, 0.0..1.0) * Color::random(rng, 0.0..1.0),
            };
            let object = Object { kind, material };
            scene.push(object);
        }
    }

    World(scene)
}

pub fn ray_info(
    rng: &mut ChaCha8Rng,
    ray: &Ray,
    world: &World,
    background: Color,
) -> (Info, Color) {
    let mut current_ray = *ray;
    let mut current_color = Color::new(1.0, 1.0, 1.0);

    enum Terminal {
        Light,
        Sky,
    }

    enum Action {
        Bounce(Color, Hit),
        Terminate(Color, Terminal),
    }

    let mut ray_color = || {
        if let Some(hit) = world.hit(&current_ray, 0.001, f64::INFINITY) {
            if let Some((attenuation, future_ray)) =
                hit.object.material.scatter(rng, &current_ray, &hit)
            {
                current_color *= attenuation;
                current_color += hit.object.material.emitted();
                current_ray = future_ray;
                Action::Bounce(current_color, hit)
            } else {
                Action::Terminate(
                    current_color * hit.object.material.emitted(),
                    Terminal::Light,
                )
            }
        } else {
            Action::Terminate(current_color * background, Terminal::Sky)
        }
    };

    let info = match ray_color() {
        Action::Bounce(current_color, hit) => Info {
            material_albedo: hit.object.material.albedo_features(),
            object_origin: hit.object.kind.origin_features(),
            hit_origin: hit.origin_features(),
            hit_normal: hit.normal_features(),
            ray_terminal: [1.0, 0.0, 0.0],
            pixel_color: [current_color.x(), current_color.y(), current_color.z()],
        },
        Action::Terminate(current_color, terminal) => {
            let mut info = Info::default();
            info.ray_terminal = match terminal {
                Terminal::Light => [0.0, 1.0, 0.0],
                Terminal::Sky => [0.0, 0.0, 1.0],
            };
            info.pixel_color = [current_color.x(), current_color.y(), current_color.z()];
            info
        }
    };

    for _ in 0..49 {
        if let Action::Terminate(current_color, _) = ray_color() {
            return (info, current_color);
        }
    }

    (info, current_color)
}

const CHANNELS: usize = 18;
const SAMPLES_PER_PIXEL: usize = 500;

#[derive(Default)]
pub struct Info {
    material_albedo: [f64; 3],
    object_origin: [f64; 3],
    hit_origin: [f64; 3],
    hit_normal: [f64; 3],
    ray_terminal: [f64; 3],
    pixel_color: [f64; 3],
}

type Features = Array3<f64>;

type Target = Array3<f64>;

#[pyclass]
struct SphereWorld {
    angles: u64,
    verticals: u64,
    size: u64,
    rng: ChaCha8Rng,
    scene: World,
}

impl SphereWorld {
    fn render_impl(&mut self, angle: u64, vertical: u64) -> (Features, Target) {
        let look_from = {
            let radius = 10.0;
            let angle_radians = angle as f64 * 2.0 * std::f64::consts::PI / self.angles as f64;
            let x = radius * angle_radians.cos();
            let z = radius * angle_radians.sin();
            Point3::new(
                x,
                (vertical as f64 + 1.0) * 9.0 / self.verticals as f64 + 1.0,
                z,
            )
        };
        self.render_at(look_from)
    }

    fn render_random_impl(&mut self) -> (Features, Target) {
        let look_from = Point3::random(&mut self.rng, -15.0..15.0);
        self.render_at(look_from)
    }

    fn render_at(&mut self, look_from: Point3) -> (Features, Target) {
        let aspect_ratio = 1.0;

        let look_at = Point3::new(0.0, 0.0, 0.0);
        let v_up = Vec3::new(0.0, 1.0, 0.0);
        let distance_to_focus = 10.0;
        let aperture = 0.1;
        let should_blur = false;
        let background = Color::new(0.0, 0.0, 0.0);

        let camera = Camera::new(
            look_from,
            look_at,
            v_up,
            20.0,
            aspect_ratio,
            aperture,
            distance_to_focus,
            should_blur,
        );

        let width = self.size as usize;
        let height = self.size as usize;

        let results: Vec<_> = iproduct!(0..height, 0..width)
            .par_bridge()
            .into_par_iter()
            .map(|(j, i)| {
                let mut rng = self.rng.clone();
                rng.set_stream((j * height + i) as u64);

                // first ray has no deviation
                let u = i as f64 / width as f64;
                let v = j as f64 / height as f64;

                let ray = camera.ray(&mut rng, u, v);
                let (pixel_info, mut pixel_color) =
                    ray_info(&mut rng, &ray, &self.scene, background);

                // whilst subsequent rays do
                for _ in 0..SAMPLES_PER_PIXEL - 1 {
                    let random_u: f64 = rng.gen();
                    let random_v: f64 = rng.gen();

                    let u = (i as f64 + random_u) / (width - 1) as f64;
                    let v = (j as f64 + random_v) / (height - 1) as f64;

                    let ray = camera.ray(&mut rng, u, v);
                    pixel_color += ray_info(&mut rng, &ray, &self.scene, background).1;
                }

                let ir = (pixel_color.x() / (SAMPLES_PER_PIXEL as f64))
                    .sqrt()
                    .min(1.0);
                let ig = (pixel_color.y() / (SAMPLES_PER_PIXEL as f64))
                    .sqrt()
                    .min(1.0);
                let ib = (pixel_color.z() / (SAMPLES_PER_PIXEL as f64))
                    .sqrt()
                    .min(1.0);

                ((j, i), (ir, ig, ib), pixel_info)
            })
            .collect();

        let mut features = Features::zeros([CHANNELS, height, width]);
        let mut target = Target::zeros([3, height, width]);

        for ((j, i), (ir, ig, ib), pixel_info) in results {
            target[[0, j, i]] = ir;
            target[[1, j, i]] = ig;
            target[[2, j, i]] = ib;

            features[[0, j, i]] = pixel_info.material_albedo[0];
            features[[1, j, i]] = pixel_info.material_albedo[1];
            features[[2, j, i]] = pixel_info.material_albedo[2];

            features[[3, j, i]] = pixel_info.object_origin[0];
            features[[4, j, i]] = pixel_info.object_origin[1];
            features[[5, j, i]] = pixel_info.object_origin[2];

            features[[6, j, i]] = pixel_info.hit_origin[0];
            features[[7, j, i]] = pixel_info.hit_origin[1];
            features[[8, j, i]] = pixel_info.hit_origin[2];

            features[[9, j, i]] = pixel_info.hit_normal[0];
            features[[10, j, i]] = pixel_info.hit_normal[1];
            features[[11, j, i]] = pixel_info.hit_normal[2];

            features[[12, j, i]] = pixel_info.ray_terminal[0];
            features[[13, j, i]] = pixel_info.ray_terminal[1];
            features[[14, j, i]] = pixel_info.ray_terminal[2];

            features[[15, j, i]] = pixel_info.pixel_color[0];
            features[[16, j, i]] = pixel_info.pixel_color[1];
            features[[17, j, i]] = pixel_info.pixel_color[2];
        }

        (features, target)
    }
}

#[pymethods]
impl SphereWorld {
    #[new]
    #[pyo3(signature = (*, angles, verticals, seed, size))]
    fn new(angles: u64, verticals: u64, seed: u64, size: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let scene = random_scene(&mut rng);
        Self {
            angles,
            verticals,
            size,
            rng,
            scene,
        }
    }

    #[pyo3(signature = (*, angle, vertical))]
    fn render<'py>(
        &mut self,
        py: Python<'py>,
        angle: u64,
        vertical: u64,
    ) -> PyResult<&'py PyTuple> {
        if angle >= self.angles {
            return Err(PyValueError::new_err(format!(
                "angle cannot be bigger than {}",
                self.angles - 1
            )));
        }
        if vertical >= self.verticals {
            return Err(PyValueError::new_err(format!(
                "vertical cannot be bigger than {}",
                self.verticals - 1
            )));
        }
        let (features, target) = self.render_impl(angle, vertical);
        Ok(PyTuple::new(
            py,
            &[features.to_pyarray(py), target.to_pyarray(py)],
        ))
    }

    #[pyo3()]
    fn render_random<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyTuple> {
        let (features, target) = self.render_random_impl();
        Ok(PyTuple::new(
            py,
            &[features.to_pyarray(py), target.to_pyarray(py)],
        ))
    }
}

#[pymodule]
fn sphere_world(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SphereWorld>()?;
    m.add("CHANNELS", CHANNELS)?;
    Ok(())
}
