use rand::Rng;

use crate::{
    hit::Hit,
    ray::Ray,
    vec3::{Color, Vec3},
};

#[derive(Clone, Copy)]
pub enum Material {
    Lambertian { albedo: Color },
    Metal { albedo: Color, fuzz: f64 },
    Dielectric { index_of_refraction: f64 },
}

impl Material {
    pub fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<(Color, Ray)> {
        match self {
            Material::Lambertian { albedo } => {
                let mut scatter_direction = hit.normal + Vec3::random_in_unit_sphere().normalized();
                if scatter_direction.near_zero() {
                    scatter_direction = hit.normal;
                }
                let scattered = Ray::new(hit.point, scatter_direction);
                Some((*albedo, scattered))
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = ray.direction.reflect(hit.normal).normalized();
                let scattered =
                    Ray::new(hit.point, reflected + *fuzz * Vec3::random_in_unit_sphere());

                if scattered.direction.dot(hit.normal) > 0.0 {
                    Some((*albedo, scattered))
                } else {
                    None
                }
            }
            Material::Dielectric {
                index_of_refraction,
            } => {
                let refraction_ratio = if hit.front_face {
                    1.0 / *index_of_refraction
                } else {
                    *index_of_refraction
                };

                let unit_direction = ray.direction.normalized();
                let cos_theta = (-1.0 * unit_direction).dot(hit.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta.powi(2)).sqrt();

                let mut rng = rand::thread_rng();
                let cannot_refract = refraction_ratio * sin_theta > 1.0;
                let will_reflect =
                    rng.gen::<f64>() < dielectric_reflectance(cos_theta, refraction_ratio);

                let direction = if cannot_refract || will_reflect {
                    unit_direction.reflect(hit.normal)
                } else {
                    unit_direction.refract(hit.normal, refraction_ratio)
                };

                let scattered = Ray::new(hit.point, direction);

                Some((Color::new(1.0, 1.0, 1.0), scattered))
            }
        }
    }
}

fn dielectric_reflectance(cosine: f64, ref_idx: f64) -> f64 {
    let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}
