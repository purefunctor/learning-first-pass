use crate::{
    ray::Ray,
    vec3::{Point3, Vec3},
};

pub struct Camera {
    look_from: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    cu: Vec3,
    cv: Vec3,
    lens_radius: f64,
    should_blur: bool,
}

impl Camera {
    pub fn new(
        look_from: Point3,
        look_at: Point3,
        v_up: Vec3,
        v_fov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_distance: f64,
        should_blur: bool,
    ) -> Self {
        // Vertical field-of-view in degrees
        let theta = std::f64::consts::PI / 180.0 * v_fov;
        let viewport_height = 2.0 * (theta / 2.0).tan();
        let viewport_width = aspect_ratio * viewport_height;

        let cw = (look_from - look_at).normalized();
        let cu = v_up.cross(cw).normalized();
        let cv = cw.cross(cu);

        let horizontal = focus_distance * viewport_width * cu;
        let vertical = focus_distance * viewport_height * cv;

        let lower_left_corner = look_from - horizontal / 2.0 - vertical / 2.0 - focus_distance * cw;
        let lens_radius = aperture / 2.0;

        Self {
            look_from,
            lower_left_corner,
            horizontal,
            vertical,
            cu,
            cv,
            lens_radius,
            should_blur,
        }
    }

    pub fn ray(&self, s: f64, t: f64) -> Ray {
        let lens_radius_random = self.lens_radius * Vec3::random_in_unit_disk();
        let offset = self.cu * lens_radius_random.x() + self.cv * lens_radius_random.y();
        Ray::new(
            self.look_from
                + if self.should_blur {
                    offset
                } else {
                    Vec3::new(0.0, 0.0, 0.0)
                },
            self.lower_left_corner + s * self.horizontal + t * self.vertical
                - self.look_from
                - if self.should_blur {
                    offset
                } else {
                    Vec3::new(0.0, 0.0, 0.0)
                },
        )
    }
}
