use camera::Camera;
use material::Material;
use ndarray::Array3;
use numpy::ToPyArray;
use pyo3::{
    pymodule,
    types::{PyModule, PyTuple},
    PyResult, Python, ToPyObject,
};
use rand::Rng;
use ray::Ray;
use vec3::{Color, Point3, Vec3};
use world::{Object, ObjectKind, World};

mod camera;
mod hit;
mod material;
mod ray;
mod vec3;
mod world;

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

pub fn ray_info(ray: &Ray, world: &World) -> (Info, Color) {
    if let Some(hit) = world.hit(ray, 0.001, f64::INFINITY) {
        if let Some((attenuation, _)) = hit.object.material.scatter(ray, &hit) {
            (Info::from_object(hit.object), attenuation)
        } else {
            (Info::from_nothing(), Color::new(0.0, 0.0, 0.0))
        }
    } else {
        let unit_direction = ray.direction.normalized();
        let t = 0.5 * (unit_direction.y() + 1.0);
        let color = (1.0 - t) * Color::new(1.0, 1.0, 1.0) + t * Color::new(0.5, 0.7, 1.0);
        (Info::from_sky(), color)
    }
}

const CHANNELS: usize = 17;

#[derive(Default)]
pub struct Info {
    is_lambertian: f64,
    is_metal: f64,
    is_dielectric: f64,
    is_nothing: f64,
    is_sky: f64,
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
}

impl Info {
    fn from_nothing() -> Self {
        let mut info = Self::default();
        info.is_nothing = 1.0;
        info
    }

    fn from_sky() -> Self {
        let mut info = Self::default();
        info.is_sky = 1.0;
        info
    }

    fn from_object(object: Object) -> Self {
        let mut info = Self::default();

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

type Features = Array3<f64>;

type Image = Array3<usize>;

pub fn generate_render(n: usize, world: World) -> (Features, Image) {
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

    let mut features = Features::zeros([CHANNELS, height, width]);
    let mut image = Image::zeros([3, height, width]);

    let mut rng = rand::thread_rng();
    for j in 0..height {
        for i in 0..width {
            // first ray has no deviation
            let u = i as f64 / (width - 1) as f64;
            let v = j as f64 / (height - 1) as f64;

            let ray = camera.ray(u, v);
            let (pixel_info, mut pixel_color) = ray_info(&ray, &world);

            // whilst subsequent rays do
            for _ in 0..SAMPLES_PER_PIXEL - 1 {
                let random_u: f64 = rng.gen();
                let random_v: f64 = rng.gen();

                let u = (i as f64 + random_u) / (width - 1) as f64;
                let v = (j as f64 + random_v) / (height - 1) as f64;

                let ray = camera.ray(u, v);
                pixel_color += ray_info(&ray, &world).1;
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

            image[[0, j, i]] = ir;
            image[[1, j, i]] = ig;
            image[[2, j, i]] = ib;

            features[[0, j, i]] = pixel_info.is_lambertian;
            features[[1, j, i]] = pixel_info.is_metal;
            features[[2, j, i]] = pixel_info.is_dielectric;
            features[[3, j, i]] = pixel_info.is_nothing;
            features[[4, j, i]] = pixel_info.is_sky;
            features[[5, j, i]] = pixel_info.lr;
            features[[6, j, i]] = pixel_info.lg;
            features[[7, j, i]] = pixel_info.lb;
            features[[8, j, i]] = pixel_info.mr;
            features[[9, j, i]] = pixel_info.mg;
            features[[10, j, i]] = pixel_info.mb;
            features[[11, j, i]] = pixel_info.mf;
            features[[12, j, i]] = pixel_info.di;
            features[[13, j, i]] = pixel_info.sphere_x;
            features[[14, j, i]] = pixel_info.sphere_y;
            features[[15, j, i]] = pixel_info.sphere_z;
            features[[16, j, i]] = pixel_info.sphere_r;
        }
    }

    (features, image)
}

#[pymodule]
fn sphere_world(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn random_scene_render(py: Python<'_>) -> &PyTuple {
        let scene = random_scene();
        let (features, image) = generate_render(128, scene);
        PyTuple::new(
            py,
            &[
                features.to_pyarray(py).to_object(py),
                image.to_pyarray(py).to_object(py),
            ],
        )
    }

    Ok(())
}
