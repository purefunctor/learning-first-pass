use std::{
    fs::{self, File},
    io::BufWriter,
    ops::{Add, Index},
    path::Path,
    sync::Arc,
};

use camera::Camera;
use clap::Parser;
use hittable::Hittable;
use hittable::World;
use indicatif::{ProgressIterator, ProgressStyle};
use material::{Dielectric, Lambertian, Metal};
use ndarray::Array3;
use num_traits::Zero;
use object::Sphere;
use rand::Rng;
use ray::Ray;
use vec3::{Color, Point3, Vec3};

mod camera;
mod hittable;
mod material;
mod object;
mod ray;
mod vec3;

fn random_scene() -> Vec<Sphere> {
    let mut rng = rand::thread_rng();
    let mut scene = Vec::new();

    for a in -1..=1 {
        for b in -1..=1 {
            let choose_mat: f64 = rng.gen();
            let sphere_size: f64 = rng.gen_range(0.49..0.5);

            let center = Point3::new(
                (a as f64) + rng.gen_range(0.0..0.9),
                0.2,
                (b as f64) + rng.gen_range(0.0..0.9),
            );

            if choose_mat < 0.50 {
                let albedo = Color::random(0.0..1.0) * Color::random(0.0..1.0);
                let sphere_mat = Arc::new(Lambertian::new(albedo));
                let sphere = Sphere::new(center, sphere_size, sphere_mat);
                scene.push(sphere);
            } else if choose_mat < 0.75 {
                let albedo = Color::random(0.4..1.0);
                let fuzz = rng.gen_range(0.0..0.5);
                let sphere_mat = Arc::new(Metal::new(albedo, fuzz));
                let sphere = Sphere::new(center, sphere_size, sphere_mat);
                scene.push(sphere)
            } else {
                let sphere_mat = Arc::new(Dielectric::new(1.5));
                let sphere = Sphere::new(center, sphere_size, sphere_mat);
                scene.push(sphere)
            }
        }
    }

    scene
}

fn compute_bounds(scene: &[Sphere]) -> (Vec3, Vec3) {
    let mut min_bounds = vec![f64::INFINITY, f64::INFINITY, f64::INFINITY];
    let mut max_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];

    for sphere in scene.iter() {
        let (x, y, z) = sphere.center.as_tuple();

        let min_coords = vec![x - sphere.radius, y - sphere.radius, z - sphere.radius];

        let max_coords = vec![x + sphere.radius, y + sphere.radius, z + sphere.radius];

        for i in 0..3 {
            min_bounds[i] = min_bounds[i].min(min_coords[i]);
            max_bounds[i] = max_bounds[i].max(max_coords[i]);
        }
    }

    assert!(min_bounds < max_bounds);

    (
        Vec3::new(min_bounds[0], min_bounds[1], min_bounds[2]),
        Vec3::new(max_bounds[0], max_bounds[1], max_bounds[2]),
    )
}

fn inside_sphere(point: Vec3, sphere: &Sphere) -> bool {
    let origin = sphere.center;
    let radius = sphere.radius;
    let distance = (point.x() - origin.x()).powi(2)
        + (point.y() - origin.y()).powi(2)
        + (point.z() - origin.z()).powi(2);
    distance <= radius.powi(2)
}

fn generate_volume(n: usize, scene: &[Sphere]) -> Vec<Vec<Vec<Vec<f64>>>> {
    let (min_bounds, max_bounds) = compute_bounds(&scene);

    let w: usize = n;
    let h: usize = n;
    let d: usize = n;

    let mut scene_ndarray: Array3<Features> = Array3::zeros((w, h, d));

    // let x_d = max_bounds.x() - min_bounds.x();
    // let y_d = max_bounds.y() - min_bounds.y();
    // let z_d = max_bounds.z() - min_bounds.z();
    // let d_m = x_d.max(y_d).max(z_d);

    // let dx = (x_d + d_m - x_d) / w as f64;
    // let dy = (y_d + d_m - y_d) / h as f64;
    // let dz = (z_d + d_m - z_d) / d as f64;

    let dx = (max_bounds.x() - min_bounds.x()) / w as f64;
    let dy = (max_bounds.y() - min_bounds.y()) / h as f64;
    let dz = (max_bounds.z() - min_bounds.z()) / d as f64;

    for ((i, j, k), v) in scene_ndarray.indexed_iter_mut() {
        let point = Vec3::new(
            min_bounds.x() + (i as f64 * dx),
            min_bounds.y() + (j as f64 * dy),
            min_bounds.z() + (k as f64 * dz),
        );
        for sphere in scene.iter() {
            if inside_sphere(point, sphere) {
                *v = Features::from_sphere(sphere);
                continue;
            }
        }
    }

    let mut scene_matrix = vec![vec![vec![vec![0.0; w]; h]; d]; 7];

    for i_c in 0..7 {
        for i_d in 0..d {
            for i_h in 0..h {
                for i_w in 0..w {
                    scene_matrix[i_c][i_d][i_h][i_w] = scene_ndarray[(i_w, i_h, i_d)][i_c];
                }
            }
        }
    }

    scene_matrix
}

fn ray_color(r: &Ray, w: &World) -> Color {
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

fn generate_render(n: usize, scene: Vec<Sphere>) -> Vec<Vec<Vec<usize>>> {
    let mut world: World = vec![];
    for sphere in scene.into_iter() {
        world.push(Box::new(sphere));
    }

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

    let mut scene_matrix = vec![vec![vec![0; height]; width]; channel];

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

            scene_matrix[0][i][j] = ir;
            scene_matrix[1][i][j] = ig;
            scene_matrix[2][i][j] = ib;
        }
    }

    scene_matrix
}

#[derive(Clone, Copy)]
struct Features {
    features: [f64; 7],
}

impl Features {
    fn from_sphere(sphere: &Sphere) -> Features {
        let mut features = [0.0; 7];

        let any_material = sphere.material.as_any();
        if let Some(material) = any_material.downcast_ref::<Lambertian>() {
            features[0] = 1.0;
            features[3] = material.albedo.x();
            features[4] = material.albedo.y();
            features[5] = material.albedo.z();
        } else if let Some(material) = any_material.downcast_ref::<Metal>() {
            features[1] = 1.0;
            features[3] = material.albedo.x();
            features[4] = material.albedo.y();
            features[5] = material.albedo.z();
            features[6] = material.fuzz;
        } else if let Some(material) = any_material.downcast_ref::<Dielectric>() {
            features[2] = 1.0;
            features[6] = material.index_of_refraction;
        } else {
            unreachable!("Unhandled material!");
        };

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

#[derive(Parser)]
struct Args {
    #[arg(help = "The sample size", default_value = "100")]
    count: usize,
    #[arg(long, help = "The output path")]
    output: String,
    #[arg(long, help = "The size of the volume", default_value = "64")]
    size: usize,
}

fn main() {
    let args = Args::parse();

    let output_path = Path::new(&args.output);
    if !output_path.exists() {
        fs::create_dir(output_path).unwrap();
    }

    for batch in (0..args.count).progress_with_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap(),
    ) {
        let scene = random_scene();
        let scene_as_volume = generate_volume(args.size, &scene);
        let scene_as_render = generate_render(args.size, scene);

        let volume_path = output_path
            .join(format!("volume_{}", batch))
            .with_extension("json");
        let render_path = output_path
            .join(format!("render_{}", batch))
            .with_extension("json");

        let volume_file = BufWriter::new(File::create(volume_path).unwrap());
        let render_file = BufWriter::new(File::create(render_path).unwrap());

        serde_json::to_writer(volume_file, &scene_as_volume).unwrap();
        serde_json::to_writer(render_file, &scene_as_render).unwrap();
    }
}
