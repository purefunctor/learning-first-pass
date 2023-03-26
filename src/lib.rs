use std::ops::{Add, Index};

use camera::Camera;
use material::Material;
use ndarray::{ArrayD, IxDyn};
use num_traits::Zero;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rand::Rng;
use ray::Ray;
use vec3::{Color, Point3, Vec3};
use world::{Object, ObjectKind, World};

mod camera;
mod material;
mod ray;
mod vec3;
mod world;
mod hit;

pub fn random_scene() -> World {
    let mut rng = rand::thread_rng();
    let mut scene = Vec::new();

    for a in -11..=11 {
        for b in -11..=11 {
            let random_material: f64 = rng.gen();

            let origin = Point3::new(
                (a as f64) + rng.gen_range(0.0..0.9),
                0.2,
                (b as f64) + rng.gen_range(0.0..0.9),
            );

            let radius: f64 = rng.gen_range(0.49..0.5);

            let kind = ObjectKind::Sphere { origin, radius };

            if random_material < 0.50 {
                let material = Material::Lambertian {
                    albedo: Color::random(0.0..1.0) * Color::random(0.0..1.0),
                };
                let object = Object { kind, material };
                scene.push(object);
            } else if random_material < 0.75 {
                let material = Material::Metal {
                    albedo: Color::random(0.4..1.0),
                    fuzz: rng.gen_range(0.0..0.5),
                };
                let object = Object { kind, material };
                scene.push(object)
            } else {
                let material = Material::Dielectric {
                    index_of_refraction: 1.5,
                };
                let object = Object { kind, material };
                scene.push(object);
            }
        }
    }

    World(scene)
}

pub fn ray_color(r: &Ray, w: &World) -> Color {
    if let Some(hit) = w.hit(r, 0.001, f64::INFINITY) {
        if let Some((attenuation, _)) = hit.material.scatter(r, &hit) {
            attenuation
        } else {
            Color::new(0.0, 0.0, 0.0)
        }
    } else {
        let unit_direction = r.direction.normalized();
        let t = 0.5 * (unit_direction.y() + 1.0);
        (1.0 - t) * Color::new(1.0, 1.0, 1.0) + t * Color::new(0.5, 0.7, 1.0)
    }
}

pub fn generate_render(n: usize, world: World) -> ArrayD<usize> {
    const SAMPLES_PER_PIXEL: usize = 50;

    let aspect_ratio = 1.0 / 1.0;
    let look_from = Point3::new(13.0, 2.0, 3.0);
    let look_at = Point3::new(0.0, 0.0, 0.0);
    let v_up = Vec3::new(0.0, 1.0, 0.0);
    let distance_to_focus = 10.0;
    let aperture = 0.1;
    let should_blur = false;

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

    let width = n;
    let height = n;
    let channel = 3;

    let mut scene_matrix = ArrayD::<usize>::zeros(IxDyn(&[height, width, channel]));

    let mut rng = rand::thread_rng();
    for j in 0..height {
        for i in 0..width {
            let mut pixel_color = Color::new(0.0, 0.0, 0.0);
            for _ in 0..SAMPLES_PER_PIXEL {
                let random_u: f64 = rng.gen();
                let random_v: f64 = rng.gen();

                let u = (i as f64 + random_u) / (width - 1) as f64;
                let v = (j as f64 + random_v) / (height - 1) as f64;

                let ray = camera.ray(u, v);
                pixel_color += ray_color(&ray, &world);
            }

            let ir: usize = (256.0
                * (pixel_color.x() / (SAMPLES_PER_PIXEL as f64))
                    .sqrt()
                    .clamp(0.0, 0.999)) as usize;
            let ig: usize = (256.0
                * (pixel_color.y() / (SAMPLES_PER_PIXEL as f64))
                    .sqrt()
                    .clamp(0.0, 0.999)) as usize;
            let ib: usize = (256.0
                * (pixel_color.z() / (SAMPLES_PER_PIXEL as f64))
                    .sqrt()
                    .clamp(0.0, 0.999)) as usize;

            scene_matrix[[j, i, 0]] = ir;
            scene_matrix[[j, i, 1]] = ig;
            scene_matrix[[j, i, 2]] = ib;
        }
    }

    scene_matrix
}

#[derive(Clone, Copy)]
pub struct Features {
    features: [f64; 7],
}

impl Features {
    pub fn from_sphere(object: &Object) -> Features {
        let mut features = [0.0; 7];

        match object.material {
            material::Material::Lambertian { albedo } => {
                features[0] = 1.0;
                features[3] = albedo.x();
                features[4] = albedo.y();
                features[5] = albedo.z();
            }
            material::Material::Metal { albedo, fuzz } => {
                features[1] = 1.0;
                features[3] = albedo.x();
                features[4] = albedo.y();
                features[5] = albedo.z();
                features[6] = fuzz;
            }
            material::Material::Dielectric {
                index_of_refraction,
            } => {
                features[2] = 1.0;
                features[6] = index_of_refraction;
            }
        }

        Self { features }
    }
}

impl Add for Features {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut accumulator = [0.0; 7];
        for (index, value) in accumulator.iter_mut().enumerate() {
            *value = self.features[index] + other.features[index]
        }
        Self {
            features: accumulator,
        }
    }
}

impl Zero for Features {
    fn zero() -> Self {
        Self { features: [0.0; 7] }
    }

    fn is_zero(&self) -> bool {
        self.features.iter().all(|v| *v == 0.0)
    }
}

impl Index<usize> for Features {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.features[index]
    }
}

#[pymodule]
fn sphere_world(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn random_scene_render(py: Python<'_>) -> &PyArrayDyn<usize> {
        let scene = random_scene();
        let render = generate_render(128, scene);
        render.to_pyarray(py)
    }

    Ok(())
}
