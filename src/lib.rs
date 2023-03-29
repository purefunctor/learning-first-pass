use std::{num::NonZeroUsize, ops::Index};

use camera::Camera;
use lru::LruCache;
use material::Material;
use ndarray::{Array3, Array4};
use numpy::ToPyArray;
use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods, pymodule,
    types::{PyModule, PyTuple},
    PyResult, Python, ToPyObject,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use ray::Ray;
use vec3::{Color, Point3, Vec3};
use world::{Object, ObjectKind, World};

mod camera;
mod hit;
mod material;
mod ray;
mod vec3;
mod world;

pub fn random_scene(rng: &mut StdRng) -> World {
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

            if random_material < 0.80 {
                let material = Material::Lambertian {
                    albedo: Color::random(rng, 0.0..1.0) * Color::random(rng, 0.0..1.0),
                };
                let object = Object { kind, material };
                scene.push(object);
            } else if random_material < 0.90 {
                let material = Material::Metal {
                    albedo: Color::random(rng, 0.4..1.0),
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

pub fn ray_color(rng: &mut StdRng, ray: &Ray, world: &World) -> Color {
    if let Some(hit) = world.hit(ray, 0.001, f64::INFINITY) {
        if let Some((attenuation, _)) = hit.object.material.scatter(rng, ray, &hit) {
            attenuation
        } else {
            Color::new(0.0, 0.0, 0.0)
        }
    } else {
        let unit_direction = ray.direction.normalized();
        let t = 0.5 * (unit_direction.y() + 1.0);
        let color = (1.0 - t) * Color::new(1.0, 1.0, 1.0) + t * Color::new(0.5, 0.7, 1.0);
        color
    }
}

const CHANNELS: usize = 19;

#[derive(Default)]
pub struct Info {
    is_lambertian: f64,
    is_metal: f64,
    is_dielectric: f64,
    is_nothing: f64,
    lr: f64,
    lg: f64,
    lb: f64,
    mr: f64,
    mg: f64,
    mb: f64,
    mf: f64,
    di: f64,
    sphere_x: f64,
    sphere_y: f64,
    sphere_z: f64,
    sphere_r: f64,
    camera_x: f64,
    camera_y: f64,
    camera_z: f64,
}

impl Info {
    fn from_nothing() -> Self {
        Self {
            is_nothing: 1.0,
            ..Default::default()
        }
    }

    fn from_object(look_from: Point3, object: &Object) -> Self {
        let mut info = Self {
            camera_x: look_from.x(),
            camera_y: look_from.y(),
            camera_z: look_from.z(),
            ..Default::default()
        };

        match object.kind {
            ObjectKind::Sphere { origin, radius } => {
                info.sphere_x = origin.x();
                info.sphere_y = origin.y();
                info.sphere_z = origin.z();
                info.sphere_r = radius;
                match object.material {
                    Material::Lambertian { albedo } => {
                        info.is_lambertian = 1.0;
                        info.lr = albedo.x();
                        info.lg = albedo.y();
                        info.lb = albedo.z();
                    }
                    Material::Metal { albedo, fuzz } => {
                        info.is_metal = 1.0;
                        info.mr = albedo.x();
                        info.mg = albedo.y();
                        info.mb = albedo.z();
                        info.mf = fuzz;
                    }
                    Material::Dielectric {
                        index_of_refraction,
                    } => {
                        info.is_dielectric = 1.0;
                        info.di = index_of_refraction;
                    }
                }
            }
        }

        info
    }
}

impl Index<usize> for Info {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.is_lambertian,
            1 => &self.is_metal,
            2 => &self.is_dielectric,
            3 => &self.is_nothing,
            4 => &self.lr,
            5 => &self.lg,
            6 => &self.lb,
            7 => &self.mr,
            8 => &self.mg,
            9 => &self.mb,
            10 => &self.mf,
            11 => &self.di,
            12 => &self.sphere_x,
            13 => &self.sphere_y,
            14 => &self.sphere_z,
            15 => &self.sphere_r,
            16 => &self.camera_x,
            17 => &self.camera_y,
            18 => &self.camera_z,
            _ => panic!("Invalid index."),
        }
    }
}

type Volume = Array4<f64>;

type Target = Array3<f64>;

#[pyclass]
struct SphereWorld {
    angles: u64,
    size: u64,
    rng: StdRng,
    scene: World,
}

impl SphereWorld {
    fn render_impl(&mut self, look_from: Point3) -> Target {
        const SAMPLES_PER_PIXEL: usize = 50;

        let aspect_ratio = 1.0;
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

        let width = self.size as usize;
        let height = self.size as usize;

        let mut target = Target::zeros([3, height, width]);

        for j in 0..height {
            for i in 0..width {
                // first ray has no deviation
                let u = i as f64 / (width - 1) as f64;
                let v = j as f64 / (height - 1) as f64;

                let ray = camera.ray(&mut self.rng, u, v);
                let mut pixel_color = ray_color(&mut self.rng, &ray, &self.scene);

                // whilst subsequent rays do
                for _ in 0..SAMPLES_PER_PIXEL - 1 {
                    let random_u: f64 = self.rng.gen();
                    let random_v: f64 = self.rng.gen();

                    let u = (i as f64 + random_u) / (width - 1) as f64;
                    let v = (j as f64 + random_v) / (height - 1) as f64;

                    let ray = camera.ray(&mut self.rng, u, v);
                    pixel_color += ray_color(&mut self.rng, &ray, &self.scene);
                }

                let ir = (pixel_color.x() / (SAMPLES_PER_PIXEL as f64)).sqrt();
                let ig = (pixel_color.y() / (SAMPLES_PER_PIXEL as f64)).sqrt();
                let ib = (pixel_color.z() / (SAMPLES_PER_PIXEL as f64)).sqrt();

                target[[0, j, i]] = ir;
                target[[1, j, i]] = ig;
                target[[2, j, i]] = ib;
            }
        }

        target
    }

    fn compute_bounds(&self) -> (Vec3, Vec3) {
        let mut min_bounds = vec![f64::INFINITY, f64::INFINITY, f64::INFINITY];
        let mut max_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];

        for object in self.scene.0.iter() {
            match object.kind {
                ObjectKind::Sphere { origin, radius } => {
                    let (x, y, z) = origin.as_tuple();

                    let min_coords = vec![x - radius, y - radius, z - radius];

                    let max_coords = vec![x + radius, y + radius, z + radius];

                    for i in 0..3 {
                        min_bounds[i] = min_bounds[i].min(min_coords[i]);
                        max_bounds[i] = max_bounds[i].max(max_coords[i]);
                    }
                }
            }
        }

        assert!(min_bounds < max_bounds);

        (
            Vec3::new(min_bounds[0], min_bounds[1], min_bounds[2]),
            Vec3::new(max_bounds[0], max_bounds[1], max_bounds[2]),
        )
    }

    fn generate_volume_impl(&self, look_from: Point3) -> Volume {
        let (min_bounds, max_bounds) = self.compute_bounds();

        let w = self.size as usize;
        let h = self.size as usize;
        let d = self.size as usize;

        let mut volume = Array4::zeros((CHANNELS, d, h, w));

        let x_d = max_bounds.x() - min_bounds.x();
        let y_d = max_bounds.y() - min_bounds.y();
        let z_d = max_bounds.z() - min_bounds.z();
        let d_m = x_d.max(y_d).max(z_d);

        let dx = (x_d + d_m - x_d) / w as f64;
        // let dy = (y_d + d_m - y_d) / h as f64;
        let dy = (max_bounds.y() - min_bounds.y()) / h as f64;
        let dz = (z_d + d_m - z_d) / d as f64;

        let mut point_cache: LruCache<(usize, usize, usize, usize), Info> =
            LruCache::new(NonZeroUsize::new(d * h * w).unwrap());

        for ((i, j, k, l), v) in volume.indexed_iter_mut() {
            let point = Vec3::new(
                min_bounds.x() + (l as f64 * dx),
                min_bounds.y() + (k as f64 * dy),
                min_bounds.z() + (j as f64 * dz),
            );
            for (index, object) in self.scene.0.iter().enumerate() {
                if let Some(info) = point_cache.get(&(j, k, l, index)) {
                    *v = info[i];
                    continue;
                }
                let info = if inside_object(point, object) {
                    Info::from_object(look_from, object)
                } else {
                    Info::from_nothing()
                };
                *v = info[i];
                point_cache.push((j, k, l, index), info);
            }
        }

        volume
    }
}

fn inside_object(point: Vec3, object: &Object) -> bool {
    match object.kind {
        ObjectKind::Sphere { origin, radius } => {
            let distance = (point.x() - origin.x()).powi(2)
                + (point.y() - origin.y()).powi(2)
                + (point.z() - origin.z()).powi(2);
            distance <= radius.powi(2)
        }
    }
}

#[pymethods]
impl SphereWorld {
    #[new]
    #[pyo3(signature = (*, angles, seed, size))]
    fn new(angles: u64, seed: u64, size: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scene = random_scene(&mut rng);
        Self {
            angles,
            size,
            rng,
            scene,
        }
    }

    #[pyo3(signature = (*, angle))]
    fn render<'py>(&mut self, py: Python<'py>, angle: u64) -> PyResult<&'py PyTuple> {
        if angle >= self.angles {
            return Err(PyValueError::new_err(format!(
                "angle cannot be bigger than {}",
                self.angles - 1
            )));
        }

        let look_from = {
            let radius = 10.0;
            let angle_radians = angle as f64 * 2.0 * std::f64::consts::PI / self.angles as f64;
            let x = radius * angle_radians.cos();
            let z = radius * angle_radians.sin();
            Point3::new(x, 10.0, z)
        };

        let target = self.render_impl(look_from);
        let volume = self.generate_volume_impl(look_from);

        Ok(PyTuple::new(
            py,
            &[
                volume.to_pyarray(py).to_object(py),
                target.to_pyarray(py).to_object(py),
            ],
        ))
    }
}

#[pymodule]
fn sphere_world(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SphereWorld>()?;
    Ok(())
}
