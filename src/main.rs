use std::sync::Arc;

use indicatif::{ProgressIterator, ProgressStyle};
use material::{Dielectric, Lambertian, Metal};
use ndarray::Array3;
use object::Sphere;
use rand::Rng;
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

    for a in -11..=11 {
        for b in -11..=11 {
            let choose_mat: f64 = rng.gen();
            let sphere_size: f64 = rng.gen_range(0.0..0.5);

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
    distance < radius
}

fn main() {
    let scene = random_scene();
    let (min_bounds, max_bounds) = compute_bounds(&scene);

    let n = 128;
    let mut scene_matrix: Array3<f64> = Array3::zeros((n, n, n));

    let dx = (max_bounds.x() - min_bounds.x()) / n as f64;
    let dy = (max_bounds.y() - min_bounds.y()) / n as f64;
    let dz = (max_bounds.z() - min_bounds.z()) / n as f64;

    for ((i, j, k), v) in scene_matrix
        .indexed_iter_mut()
        .progress_with_style(ProgressStyle::default_bar())
    {
        let point = Vec3::new(
            min_bounds.x() + (i as f64 * dx),
            min_bounds.y() + (j as f64 * dy),
            min_bounds.z() + (k as f64 * dz),
        );
        for sphere in scene.iter() {
            if inside_sphere(point, sphere) {
                *v = 1.0;
                continue;
            }
        }
    }

    println!("{:?}", scene_matrix.shape());
}
