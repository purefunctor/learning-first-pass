use std::any::Any;

use rand::Rng;

use crate::{
    hittable::Hit,
    ray::Ray,
    vec3::{Color, Vec3},
};

pub trait Scatter: Send + Sync {
    fn scatter(&self, r_in: &Ray, hit: &Hit) -> Option<(Color, Ray)>;

    fn as_any(&self) -> &dyn Any;
}

pub struct Lambertian {
    pub albedo: Color,
}

impl Lambertian {
    pub fn new(albedo: Color) -> Self {
        Self { albedo }
    }
}

impl Scatter for Lambertian {
    fn scatter(&self, _: &Ray, hit: &Hit) -> Option<(Color, Ray)> {
        let mut scatter_direction = hit.normal + Vec3::random_in_unit_sphere().normalized();
        if scatter_direction.near_zero() {
            scatter_direction = hit.normal;
        }
        let scattered = Ray::new(hit.point, scatter_direction);
        Some((self.albedo, scattered))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct Metal {
    pub albedo: Color,
    pub fuzz: f64,
}

impl Metal {
    pub fn new(albedo: Color, fuzz: f64) -> Self {
        Self { albedo, fuzz }
    }
}

impl Scatter for Metal {
    fn scatter(&self, r_in: &Ray, hit: &Hit) -> Option<(Color, Ray)> {
        let reflected = r_in.direction.reflect(hit.normal).normalized();
        let scattered = Ray::new(
            hit.point,
            reflected + self.fuzz * Vec3::random_in_unit_sphere(),
        );

        if scattered.direction.dot(hit.normal) > 0.0 {
            Some((self.albedo, scattered))
        } else {
            None
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct Dielectric {
    pub index_of_refraction: f64,
}

impl Dielectric {
    pub fn new(index_of_refraction: f64) -> Dielectric {
        Dielectric {
            index_of_refraction,
        }
    }

    fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
        let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Scatter for Dielectric {
    fn scatter(&self, r_in: &Ray, hit: &Hit) -> Option<(Color, Ray)> {
        let refraction_ratio = if hit.front_face {
            1.0 / self.index_of_refraction
        } else {
            self.index_of_refraction
        };

        let unit_direction = r_in.direction.normalized();
        let cos_theta = (-1.0 * unit_direction).dot(hit.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta.powi(2)).sqrt();

        let mut rng = rand::thread_rng();
        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let will_reflect = rng.gen::<f64>() < Self::reflectance(cos_theta, refraction_ratio);

        let direction = if cannot_refract || will_reflect {
            unit_direction.reflect(hit.normal)
        } else {
            unit_direction.refract(hit.normal, refraction_ratio)
        };

        let scattered = Ray::new(hit.point, direction);

        Some((Color::new(1.0, 1.0, 1.0), scattered))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
