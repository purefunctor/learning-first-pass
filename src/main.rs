/*r
3. Materials (diffuse, metal, dielectric)

4. Objects (sphere)

5. World (vec of objects)

6. Hittable stuff

 */

use std::{
    io::{stderr, Write},
    rc::Rc,
};

use rand::Rng;

use hittable::{Hittable, World};
use material::{Dielectric, Lambertian, Metal};
use object::Sphere;
use ray::Ray;
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

    let ground_mat = Rc::new(Lambertian::new(Color::new(0.5, 0.5, 0.5)));
    let ground_sphere = Sphere::new(Point3::new(0.0, -1000.0, 0.0), 1000.0, ground_mat);

    world.push(Box::new(ground_sphere));

    for a in -11..=11 {
        for b in -11..=11 {
            let choose_mat: f64 = rng.gen();
            let center = Point3::new(
                (a as f64) + rng.gen_range(0.0..0.9),
                0.2,
                (b as f64) + rng.gen_range(0.0..0.9),
            );

            if choose_mat < 0.8 {
                // Diffuse
                let albedo = Color::random(0.0..1.0) * Color::random(0.0..1.0);
                let sphere_mat = Rc::new(Lambertian::new(albedo));
                let sphere = Sphere::new(center, 0.2, sphere_mat);

                world.push(Box::new(sphere));
            } else if choose_mat < 0.95 {
                // Metal
                let albedo = Color::random(0.4..1.0);
                let fuzz = rng.gen_range(0.0..0.5);
                let sphere_mat = Rc::new(Metal::new(albedo, fuzz));
                let sphere = Sphere::new(center, 0.2, sphere_mat);

                world.push(Box::new(sphere));
            } else {
                // Glass
                let sphere_mat = Rc::new(Dielectric::new(1.5));
                let sphere = Sphere::new(center, 0.2, sphere_mat);

                world.push(Box::new(sphere));
            }
        }
    }

    let mat1 = Rc::new(Dielectric::new(1.5));
    let mat2 = Rc::new(Lambertian::new(Color::new(0.4, 0.2, 0.1)));
    let mat3 = Rc::new(Metal::new(Color::new(0.7, 0.6, 0.5), 0.0));

    let sphere1 = Sphere::new(Point3::new(0.0, 1.0, 0.0), 1.0, mat1);
    let sphere2 = Sphere::new(Point3::new(-4.0, 1.0, 0.0), 1.0, mat2);
    let sphere3 = Sphere::new(Point3::new(4.0, 1.0, 0.0), 1.0, mat3);

    world.push(Box::new(sphere1));
    world.push(Box::new(sphere2));
    world.push(Box::new(sphere3));

    world
}

// fn ray_color(r: &Ray, w: &World, depth: usize) -> Color {
//     if depth <= 0 {
//         return Color::new(0.0, 0.0, 0.0);
//     }

//     if let Some(hit) = w.hit(r, 0.001, f64::INFINITY) {
//         if let Some((attenuation, scattered)) = hit.material.scatter(r, &hit) {
//             attenuation * ray_color(&scattered, w, depth - 1)
//         } else {
//             Color::new(0.0, 0.0, 0.0)
//         }
//     } else {
//         let unit_direction = r.direction.normalized();
//         let t = 0.5 * (unit_direction.y() + 1.0);
//         (1.0 - t) * Color::new(1.0, 1.0, 1.0) + t * Color::new(0.5, 0.7, 1.0)
//     }
// }

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

fn main() {
    // Image
    const ASPECT_RATIO: f64 = 1.0 / 1.0;
    const IMAGE_WIDTH: usize = 256;
    const IMAGE_HEIGHT: usize = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as usize;
    const SAMPLES_PER_PIXEL: usize = 8;
    // const MAX_DEPTH: usize = 50;

    // World
    let w = random_scene();

    // Camera
    let look_from = Point3::new(13.0, 2.0, 3.0);
    let look_at = Point3::new(0.0, 0.0, 0.0);
    let v_up = Vec3::new(0.0, 1.0, 0.0);
    let distance_to_focus = 10.0;
    let aperture = 0.1;
    let should_blur = true;

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

    println!("P3");
    println!("{} {}", IMAGE_WIDTH, IMAGE_HEIGHT);
    println!("255");

    let mut rng = rand::thread_rng();
    for j in (0..IMAGE_HEIGHT).rev() {
        eprint!("\rScanlines remaining {:?}", IMAGE_HEIGHT - j - 1);
        stderr().flush().unwrap();

        for i in 0..IMAGE_WIDTH {
            let mut pixel_color = Color::new(0.0, 0.0, 0.0);
            for _ in 0..SAMPLES_PER_PIXEL {
                let random_u: f64 = rng.gen();
                let random_v: f64 = rng.gen();

                let u: f64 = (i as f64 + random_u) / (IMAGE_WIDTH - 1) as f64;
                let v: f64 = (j as f64 + random_v) / (IMAGE_HEIGHT - 1) as f64;

                let r = camera.ray(u, v);
                // pixel_color += ray_color(&r, &world, MAX_DEPTH);
                pixel_color += ray_color_no_scatter(&r, &w);
            }
            println!("{}", pixel_color.format_color(SAMPLES_PER_PIXEL));
        }
    }
    eprintln!("\nDone.");

    println!("Hello, world!");
}
