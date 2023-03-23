use std::{
    fs::{self, File},
    io::{stderr, BufWriter, Write},
    path::Path,
    sync::Arc,
};

use clap::Parser;
use rand::Rng;

use hittable::{Hittable, World};
use material::{Dielectric, Lambertian, Metal};
use object::Sphere;
use ray::Ray;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use vec3::{Color, Point3};

use crate::{camera::Camera, vec3::Vec3};

mod camera;
mod hittable;
mod material;
mod object;
mod ray;
mod vec3;

fn random_scene() -> World {
    let mut rng = rand::thread_rng();
    let mut world = World::new();

    // let ground_mat = Arc::new(Lambertian::new(Color::new(0.5, 0.5, 0.5)));
    // let ground_sphere = Sphere::new(Point3::new(0.0, -1000.0, 0.0), 1000.0, ground_mat);

    // world.push(Box::new(ground_sphere));

    for a in -11..=11 {
        for b in -11..=11 {
            let choose_mat: f64 = rng.gen();
            let sphere_size: f64 = rng.gen_range(0.0..0.5);

            let center = Point3::new(
                (a as f64) + rng.gen_range(0.0..0.9),
                0.2,
                (b as f64) + rng.gen_range(0.0..0.9),
            );

            if choose_mat < 0.8 {
                // Diffuse
                let albedo = Color::random(0.0..1.0) * Color::random(0.0..1.0);
                let sphere_mat = Arc::new(Lambertian::new(albedo));
                let sphere = Sphere::new(center, sphere_size, sphere_mat);

                world.push(Box::new(sphere));
            } else if choose_mat < 0.95 {
                // Metal
                let albedo = Color::random(0.4..1.0);
                let fuzz = rng.gen_range(0.0..0.5);
                let sphere_mat = Arc::new(Metal::new(albedo, fuzz));
                let sphere = Sphere::new(center, sphere_size, sphere_mat);

                world.push(Box::new(sphere));
            } else {
                // Glass
                let sphere_mat = Arc::new(Dielectric::new(1.5));
                let sphere = Sphere::new(center, sphere_size, sphere_mat);

                world.push(Box::new(sphere));
            }
        }
    }

    let mat1 = Arc::new(Dielectric::new(1.5));
    let mat2 = Arc::new(Lambertian::new(Color::new(0.4, 0.2, 0.1)));
    let mat3 = Arc::new(Metal::new(Color::new(0.7, 0.6, 0.5), 0.0));

    let sphere1 = Sphere::new(Point3::new(0.0, 1.0, 0.0), 1.0, mat1);
    let sphere2 = Sphere::new(Point3::new(-4.0, 1.0, 0.0), 1.0, mat2);
    let sphere3 = Sphere::new(Point3::new(4.0, 1.0, 0.0), 1.0, mat3);

    world.push(Box::new(sphere1));
    world.push(Box::new(sphere2));
    world.push(Box::new(sphere3));

    world
}

#[allow(dead_code)]
fn ray_color(r: &Ray, w: &World, depth: usize) -> Color {
    if depth <= 0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    if let Some(hit) = w.hit(r, 0.001, f64::INFINITY) {
        if let Some((attenuation, scatter)) = hit.material.scatter(r, &hit) {
            attenuation * ray_color(&scatter, w, depth - 1)
        } else {
            Color::new(0.0, 0.0, 0.0)
        }
    } else {
        let unit_direction = r.direction.normalized();
        let t = 0.5 * (unit_direction.y() + 1.0);
        (1.0 - t) * Color::new(1.0, 1.0, 1.0) + t * Color::new(0.5, 0.7, 1.0)
    }
}

#[allow(dead_code)]
fn ray_color_no_scatter(r: &Ray, w: &World) -> Color {
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

fn ray_color_as_output(r: &Ray, w: &World) -> (Color, [f64; 9]) {
    if let Some(hit) = w.hit(r, 0.001, f64::INFINITY) {
        if let Some((attenuation, _)) = hit.material.scatter(r, &hit) {
            if let Some(_) = hit.material.as_any().downcast_ref::<Lambertian>() {
                return (
                    attenuation,
                    [
                        // is_lambertian
                        1.0,
                        // is_metal
                        0.0,
                        // is_dielectric
                        0.0,
                        // is_nothing
                        0.0,
                        // is_sky
                        0.0,
                        // lr
                        attenuation.x(),
                        // lg
                        attenuation.y(),
                        // lb
                        attenuation.z(),
                        // mr
                        0.0,
                    ],
                );
            }

            if let Some(material) = hit.material.as_any().downcast_ref::<Metal>() {
                return (
                    attenuation,
                    [
                        // is_lambertian
                        0.0,
                        // is_metal
                        1.0,
                        // is_dielectric
                        0.0,
                        // is_nothing
                        0.0,
                        // is_sky
                        0.0,
                        // mr
                        attenuation.x(),
                        // mg
                        attenuation.y(),
                        // mb
                        attenuation.z(),
                        // mf
                        material.fuzz,
                    ],
                );
            }

            if let Some(material) = hit.material.as_any().downcast_ref::<Dielectric>() {
                return (
                    attenuation,
                    [
                        // is_lambertian
                        0.0,
                        // is_metal
                        0.0,
                        // is_dielectric
                        1.0,
                        // is_nothing
                        0.0,
                        // is_sky
                        0.0,
                        // dr
                        attenuation.x(),
                        // dg
                        attenuation.y(),
                        // db
                        attenuation.z(),
                        // di
                        material.index_of_refraction,
                    ],
                );
            }

            unreachable!()
        } else {
            return (
                Color::new(0.0, 0.0, 0.0),
                [
                    // is_lambertian
                    0.0,
                    // is_metal
                    0.0,
                    // is_dielectric
                    0.0,
                    // is_nothing
                    1.0,
                    // is_sky
                    0.0,
                    // lr
                    0.0,
                    // lg
                    0.0,
                    // lb
                    0.0,
                    // mr
                    0.0,
                ],
            );
        }
    } else {
        let unit_direction = r.direction.normalized();
        let t = 0.5 * (unit_direction.y() + 1.0);
        let color = (1.0 - t) * Color::new(1.0, 1.0, 1.0) + t * Color::new(0.5, 0.7, 1.0);
        return (
            color,
            [
                // is_lambertian
                0.0,
                // is_metal
                0.0,
                // is_dielectric
                0.0,
                // is_nothing
                0.0,
                // is_sky
                1.0,
                // sr
                color.x(),
                // sg
                color.y(),
                // sb
                color.z(),
                // i
                0.0,
            ],
        );
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(help = "Start a batch from this index", default_value = "0")]
    batch_start: usize,
    #[arg(help = "Generate a batch until this index", default_value = "100")]
    batch_end: usize,
    #[arg(long, help = "The output path.")]
    output_path: String,
}

fn main() {
    let cli = Args::parse();

    let output_path = Path::new(&cli.output_path);
    if !output_path.exists() {
        fs::create_dir(output_path).unwrap();
    }

    // Image
    const ASPECT_RATIO: f64 = 1.0 / 1.0;
    const IMAGE_WIDTH: usize = 64;
    const IMAGE_HEIGHT: usize = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as usize;
    const SAMPLES_PER_PIXEL: usize = 10;
    // const MAX_DEPTH: usize = 50;

    for index in cli.batch_start..cli.batch_end {
        // World
        let w = random_scene();

        // Camera
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
            ASPECT_RATIO,
            aperture,
            distance_to_focus,
            should_blur,
        );

        let mut scanlines = vec![];
        for j in (0..IMAGE_HEIGHT).rev() {
            eprint!(
                "\r{} - Scanlines remaining {:?}",
                index,
                IMAGE_HEIGHT - j - 1
            );
            stderr().flush().unwrap();
            let scanline: Vec<_> = (0..IMAGE_WIDTH)
                .into_par_iter()
                .map(|i| {
                    let mut rng = rand::thread_rng();

                    let random_u: f64 = rng.gen();
                    let random_v: f64 = rng.gen();

                    let u = (i as f64 + random_u) / (IMAGE_WIDTH - 1) as f64;
                    let v = (j as f64 + random_v) / (IMAGE_HEIGHT - 1) as f64;

                    let r = camera.ray(u, v);
                    let (mut image_color, input_image) = ray_color_as_output(&r, &w);
                    for _ in 0..SAMPLES_PER_PIXEL - 1 {
                        let random_u: f64 = rng.gen();
                        let random_v: f64 = rng.gen();

                        let u = (i as f64 + random_u) / (IMAGE_WIDTH - 1) as f64;
                        let v = (j as f64 + random_v) / (IMAGE_HEIGHT - 1) as f64;

                        let r = camera.ray(u, v);
                        image_color += ray_color_no_scatter(&r, &w);
                    }
                    (image_color, input_image)
                })
                .collect();
            scanlines.push(scanline);
        }

        let base_file_path = output_path.join(format!("image_{}", index));
        let json_file_path = base_file_path.with_extension("json");
        let ppm_file_path = base_file_path.with_extension("ppm");

        let mut ppm_file = BufWriter::new(File::create(ppm_file_path).unwrap());
        writeln!(ppm_file, "P3").unwrap();
        writeln!(ppm_file, "{} {}", IMAGE_WIDTH, IMAGE_HEIGHT).unwrap();
        writeln!(ppm_file, "255").unwrap();
        for i in 0..IMAGE_HEIGHT {
            for j in 0..IMAGE_WIDTH {
                writeln!(ppm_file, "{}", scanlines[i][j].0.format_color(SAMPLES_PER_PIXEL)).unwrap();
            }
        }

        const CHANNEL_SIZE: usize = 9;
        let json_file = BufWriter::new(File::create(json_file_path).unwrap());
        let mut image_ndarray = vec![vec![vec![0.0; IMAGE_HEIGHT]; IMAGE_WIDTH]; CHANNEL_SIZE];
        for i in 0..IMAGE_HEIGHT {
            for j in 0..IMAGE_WIDTH {
                for k in 0..CHANNEL_SIZE {
                    image_ndarray[k][i][j] = scanlines[i][j].1[k]
                }
            }
        }
        serde_json::to_writer(json_file, &image_ndarray).unwrap();

        eprintln!(" - Done!");
    }
}
