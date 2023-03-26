use crate::{
    hittable::{Hit, Hittable},
    material::Material,
    ray::Ray,
    vec3::{Point3, Vec3},
};

pub struct Sphere {
    pub radius: f64,
    pub center: Point3,
    pub material: Material,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let oc = r.origin - self.center;
        let a = r.direction.length().powi(2);
        let half_b = oc.dot(r.direction);
        let c = oc.length().powi(2) - self.radius.powi(2);

        let discriminant = half_b.powi(2) - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            }
        }

        let point = r.at(root);
        let mut hit = Hit {
            t: root,
            material: self.material,
            normal: Vec3::new(0.0, 0.0, 0.0),
            front_face: false,
            point,
        };

        let outward_normal = (hit.point - self.center) / self.radius;
        hit.set_face_normal(r, outward_normal);

        Some(hit)
    }
}
