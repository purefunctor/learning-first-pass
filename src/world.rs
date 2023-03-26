use crate::{
    material::Material,
    ray::Ray,
    vec3::Vec3, hit::Hit,
};

pub enum ObjectKind {
    Sphere { origin: Vec3, radius: f64 },
}

pub struct Object {
    pub kind: ObjectKind,
    pub material: Material,
}

impl Object {
    pub fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        match self.kind {
            ObjectKind::Sphere { origin, radius } => {
                let oc = ray.origin - origin;
                let a = ray.direction.length().powi(2);
                let half_b = oc.dot(ray.direction);
                let c = oc.length().powi(2) - radius.powi(2);

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

                let point = ray.at(root);
                let mut hit = Hit {
                    t: root,
                    material: self.material,
                    normal: Vec3::new(0.0, 0.0, 0.0),
                    front_face: false,
                    point,
                };

                let outward_normal = (hit.point - origin) / radius;
                hit.set_face_normal(ray, outward_normal);

                Some(hit)
            }
        }
    }
}

pub struct World(pub Vec<Object>);

impl World {
    pub fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let mut current_hit = None;
        let mut closest_so_far = t_max;

        for object in self.0.iter() {
            if let Some(hit) = object.hit(ray, t_min, closest_so_far) {
                closest_so_far = hit.t;
                current_hit = Some(hit);
            }
        }

        current_hit
    }
}
