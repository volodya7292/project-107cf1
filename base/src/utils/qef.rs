// Quadratic error function

use common::nalgebra as na;
use std::ops::AddAssign;

const SVD_NUM_SWEEPS: u32 = 5;
const PSEUDO_INVERSE_THRESHOLD: f32 = 0.001;

fn mat_sym_mul_v(m: &na::Matrix3<f32>, v: &na::Vector3<f32>) -> na::Vector3<f32> {
    na::Vector3::new(
        m.column(0).dot(&v),
        m[(1, 0)] * v.x + m[(1, 1)] * v.y + m[(2, 1)] * v.z,
        m[(2, 0)] * v.x + m[(2, 1)] * v.y + m[(2, 2)] * v.z,
    )
}

fn givens_coeffs_sym(app: f32, apq: f32, aqq: f32) -> (f32, f32) {
    if apq == 0.0 {
        return (1.0, 0.0);
    }

    let tau = (aqq - app) / (apq * 2.0);
    let stt = (1.0 + tau * tau).sqrt();
    let tan = 1.0 / (tau + stt * tau.signum());

    let c = 1.0 / (1.0 + tan * tan).sqrt();
    let s = tan * c;

    (c, s)
}

fn rotate_q_xy(x: &mut f32, y: &mut f32, a: f32, c: f32, s: f32) {
    let cc = c * c;
    let ss = s * s;
    let mx = a * c * s * 2.0;
    let (u, v) = (*x, *y);

    *x = cc * u - mx + ss * v;
    *y = ss * u + mx + cc * v;
}

fn rotate_xy(x: &mut f32, y: &mut f32, c: f32, s: f32) {
    let (u, v) = (*x, *y);

    *x = c * u - s * v;
    *y = s * u + c * v;
}

fn rotate(vtav: &mut na::Matrix3<f32>, v: &mut na::Matrix3<f32>, a: usize, b: usize) {
    if vtav[(b, a)] == 0.0 {
        return;
    }

    let (c, s) = givens_coeffs_sym(vtav[(a, a)], vtav[(b, a)], vtav[(b, b)]);

    let mut x = vtav[(a, a)];
    let mut y = vtav[(b, b)];
    rotate_q_xy(&mut x, &mut y, vtav[(b, a)], c, s);
    vtav[(a, a)] = x;
    vtav[(b, b)] = y;

    let mut x = vtav[(3 - b, 0)];
    let mut y = vtav[(2, 1 - a)];
    rotate_xy(&mut x, &mut y, c, s);
    vtav[(3 - b, 0)] = x;
    vtav[(2, 1 - a)] = y;

    vtav[(b, a)] = 0.0;

    let mut x = v[(a, 0)];
    let mut y = v[(b, 0)];
    rotate_xy(&mut x, &mut y, c, s);
    v[(a, 0)] = x;
    v[(b, 0)] = y;

    let mut x = v[(a, 1)];
    let mut y = v[(b, 1)];
    rotate_xy(&mut x, &mut y, c, s);
    v[(a, 1)] = x;
    v[(b, 1)] = y;

    let mut x = v[(a, 2)];
    let mut y = v[(b, 2)];
    rotate_xy(&mut x, &mut y, c, s);
    v[(a, 2)] = x;
    v[(b, 2)] = y;
}

fn solve_sym(a: &na::Matrix3<f32>) -> (na::Matrix3<f32>, na::Vector3<f32>) {
    let mut v = na::Matrix3::identity();
    let mut vtav = *a;

    for _ in 0..SVD_NUM_SWEEPS {
        rotate(&mut vtav, &mut v, 0, 1);
        rotate(&mut vtav, &mut v, 0, 2);
        rotate(&mut vtav, &mut v, 1, 2);
    }

    (v, na::Vector3::new(vtav[(0, 0)], vtav[(1, 1)], vtav[(2, 2)]))
}

fn inv_det(x: f32) -> f32 {
    let ovx = 1.0 / x;
    if x.abs() < PSEUDO_INVERSE_THRESHOLD || ovx.abs() < PSEUDO_INVERSE_THRESHOLD {
        0.0
    } else {
        ovx
    }
}

fn pseudo_inverse(sigma: &na::Vector3<f32>, v: &na::Matrix3<f32>) -> na::Matrix3<f32> {
    let d = na::Vector3::new(inv_det(sigma[0]), inv_det(sigma[1]), inv_det(sigma[2]));

    let dv0 = d.component_mul(&v.column(0));
    let dv1 = d.component_mul(&v.column(1));
    let dv2 = d.component_mul(&v.column(2));

    let c0 = na::Vector3::new(
        dv0.dot(&v.column(0)),
        dv0.dot(&v.column(1)),
        dv0.dot(&v.column(2)),
    );
    let c1 = na::Vector3::new(
        dv1.dot(&v.column(0)),
        dv1.dot(&v.column(1)),
        dv1.dot(&v.column(2)),
    );
    let c2 = na::Vector3::new(
        dv2.dot(&v.column(0)),
        dv2.dot(&v.column(1)),
        dv2.dot(&v.column(2)),
    );

    na::Matrix3::from_columns(&[c0, c1, c2])
}

fn solve_ata_atb(ata: &na::Matrix3<f32>, atb: &na::Vector3<f32>) -> na::Vector3<f32> {
    let (v, sigma) = solve_sym(&ata);
    let v_inv = pseudo_inverse(&sigma, &v);
    return v_inv * atb;
}

/// Input: vertices [(pos, normal)]
///
/// Output: (solved position, error)
pub fn solve(vertices: &[(na::Vector3<f32>, na::Vector3<f32>)]) -> (na::Vector3<f32>, f32) {
    let mut ata = na::Matrix3::from_element(0.0);
    let mut atb = na::Vector3::from_element(0.0);
    let mut point_accum = na::Vector4::from_element(0.0);

    for vertex in vertices {
        let pos = &vertex.0;
        let normal = &vertex.1;

        ata.column_mut(0).add_assign(normal * normal.x);
        ata[(1, 1)] += normal.y * normal.y;
        ata[(2, 1)] += normal.y * normal.z;
        ata[(2, 2)] += normal.z * normal.z;

        atb += normal * pos.dot(normal);
        point_accum += na::Vector4::new(pos.x, pos.y, pos.z, 1.0);
    }

    let mass_point = point_accum.xyz() / point_accum.w;
    let atb2 = atb - mat_sym_mul_v(&ata, &mass_point);
    let x = solve_ata_atb(&ata, &atb2);

    let vtmp = atb - mat_sym_mul_v(&ata, &x);
    let error = vtmp.dot(&vtmp);

    (x + mass_point, error)
}
