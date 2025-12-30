use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba, Pixel, Primitive};
use imageproc::geometric_transformations::Interpolation;
use ndarray::{Array2, Array1, s};
use ndarray::linalg::Dot;

/// Standard 5 landmarks for 112x112 input (ArcFace)
const ARCFACE_DST: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Estimate Similarity Transform matrix (2x3) using Least Squares.
/// src: 5x2 array of source landmarks
/// dst: 5x2 array of destination landmarks
fn estimate_norm(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> [f32; 6] {
    // We want to find M such that src * M ~= dst (in homogenous coords)
    // Actually, Similarity Transform is:
    // [a  b  tx]
    // [-b a  ty]

    // Using Umeyama or simple least squares for similarity.
    // For 5 points, we can solve a linear system.
    // u = a*x - b*y + tx
    // v = b*x + a*y + ty

    // Construct A matrix (10x4) and B vector (10x1)
    // A = [x0 -y0 1 0]  B = [u0]
    //     [y0  x0 0 1]      [v0]
    //     ...

    let mut a_data = Vec::with_capacity(20);
    let mut b_data = Vec::with_capacity(10);

    for i in 0..5 {
        let x = src[i][0];
        let y = src[i][1];
        let u = dst[i][0];
        let v = dst[i][1];

        // Eq 1: a*x - b*y + tx = u
        a_data.extend_from_slice(&[x, -y, 1.0, 0.0]);
        b_data.push(u);

        // Eq 2: a*y + b*x + ty = v
        a_data.extend_from_slice(&[y, x, 0.0, 1.0]);
        b_data.push(v);
    }

    // We need to solve A * X = B for X = [a, b, tx, ty]
    // Use ndarray or just manual computation since it's small (10 equations).
    // Using Normal Equations: (A^T A) X = A^T B  => X = (A^T A)^-1 A^T B

    let a_mat = Array2::from_shape_vec((10, 4), a_data).unwrap();
    let b_vec = Array1::from_vec(b_data);

    // A^T
    let at_mat = a_mat.t();

    // A^T * A (4x4)
    let ata = at_mat.dot(&a_mat);

    // A^T * B (4x1)
    let atb = at_mat.dot(&b_vec);

    // Inverse of 4x4 matrix ATA.
    // Since we don't have a full linear algebra crate like ndarray-linalg (which requires LAPACK),
    // and the matrix is small 4x4, we can implement 4x4 inversion or Gaussian elimination.
    // However, for this specific problem (similarity transform), there is a closed form solution
    // involving means and variances if we want to be fancy, but 4x4 inverse is safe enough.

    // Let's implement Gaussian elimination for 4x4.
    let x = solve_linear_system_4x4(ata, atb);

    let a = x[0];
    let b = x[1];
    let tx = x[2];
    let ty = x[3];

    // Result matrix 2x3
    // [a  b  tx]
    // [-b a  ty]
    // Note: imageproc warp_affine expects matrix in [a, b, c, d, e, f] for
    // x' = ax + by + c
    // y' = dx + ey + f
    // So we need:
    // x' = a*x - b*y + tx  => a=a, b=-b, c=tx
    // y' = b*x + a*y + ty  => d=b, e=a, f=ty

    [a, -b, tx, b, a, ty]
}

fn solve_linear_system_4x4(a: Array2<f32>, b: Array1<f32>) -> Vec<f32> {
    // Gaussian elimination
    let mut m = a.clone();
    let mut v = b.clone();
    let n = 4;

    for i in 0..n {
        // Pivot
        let mut max_row = i;
        for k in i+1..n {
            if m[[k, i]].abs() > m[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap
        for k in i..n {
            let tmp = m[[i, k]];
            m[[i, k]] = m[[max_row, k]];
            m[[max_row, k]] = tmp;
        }
        let tmp = v[i];
        v[i] = v[max_row];
        v[max_row] = tmp;

        // Eliminate
        let pivot = m[[i, i]];
        if pivot.abs() < 1e-6 {
             // Singular matrix, return zeros or handle error
             return vec![0.0; 4];
        }

        for k in i+1..n {
            let factor = m[[k, i]] / pivot;
            m[[k, i]] = 0.0;
            for j in i+1..n {
                m[[k, j]] -= factor * m[[i, j]];
            }
            v[k] -= factor * v[i];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i+1..n {
            sum += m[[i, j]] * x[j];
        }
        x[i] = (v[i] - sum) / m[[i, i]];
    }

    x
}

/// Align face using Similarity Transform.
/// image: Source image
/// landmarks: 5 landmarks [x1, y1, ... x5, y5] (flattened)
/// Returns: 112x112 aligned face image
pub fn align_face(image: &DynamicImage, landmarks: &[f32; 10]) -> DynamicImage {
    // Convert flattened landmarks to [[f32; 2]; 5]
    let mut src_marks = [[0.0; 2]; 5];
    for i in 0..5 {
        src_marks[i][0] = landmarks[i * 2];
        src_marks[i][1] = landmarks[i * 2 + 1];
    }

    // Calculate transformation matrix
    // Note: The estimate_norm calculates M such that Src * M -> Dst
    // But warp_affine typically performs Inverse Mapping (for each pixel in Dst, find Src).
    // So we usually need the inverse transform or feed the inverse matrix.
    // Let's check imageproc::warp_affine documentation or behavior.
    // "The transformation is applied to the coordinates of the input image..."
    // Usually standard warp takes the Inverse Matrix (Dst -> Src).

    // Let's compute Src -> Dst matrix M.
    // u = M * x
    // If warp_affine iterates over destination pixels u, it needs x = M^-1 * u.
    // So we need to compute the transform from Dst TO Src.
    // Let's use estimate_norm(dst, src) instead.

    let matrix_arr = estimate_norm(&ARCFACE_DST, &src_marks);

    // imageproc uses a 2x3 matrix [a, b, c, d, e, f]
    // where x_src = a*x_dst + b*y_dst + c
    //       y_src = d*x_dst + e*y_dst + f

    let affine = [
        matrix_arr[0], matrix_arr[1], matrix_arr[2],
        matrix_arr[3], matrix_arr[4], matrix_arr[5]
    ];

    // Convert to Rgba8 for processing
    let img_buffer = image.to_rgba8();

    // Let's implement a simple affine warp for fixed size 112x112.
    // x_src = a*x + b*y + c
    // y_src = d*x + e*y + f

    let (width, height) = (112, 112);
    let mut out_img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width, height);

    let (in_w, in_h) = image.dimensions();

    for y in 0..height {
        for x in 0..width {
            let x_f = x as f32;
            let y_f = y as f32;

            let src_x = affine[0] * x_f + affine[1] * y_f + affine[2];
            let src_y = affine[3] * x_f + affine[4] * y_f + affine[5];

            // Bilinear interpolation
            let pixel = interpolate_bilinear(&img_buffer, src_x, src_y, in_w, in_h);
            out_img.put_pixel(x, y, pixel);
        }
    }

    DynamicImage::ImageRgba8(out_img)
}

fn interpolate_bilinear(img: &ImageBuffer<Rgba<u8>, Vec<u8>>, x: f32, y: f32, w: u32, h: u32) -> Rgba<u8> {
    if x < 0.0 || x >= (w as f32 - 1.0) || y < 0.0 || y >= (h as f32 - 1.0) {
        return Rgba([0, 0, 0, 0]);
    }

    let x_0 = x.floor() as u32;
    let y_0 = y.floor() as u32;
    let x_1 = x_0 + 1;
    let y_1 = y_0 + 1;

    let dx = x - x_0 as f32;
    let dy = y - y_0 as f32;

    let p00 = img.get_pixel(x_0, y_0).0;
    let p10 = img.get_pixel(x_1, y_0).0;
    let p01 = img.get_pixel(x_0, y_1).0;
    let p11 = img.get_pixel(x_1, y_1).0;

    let mut out = [0u8; 4];
    for i in 0..4 {
        let v00 = p00[i] as f32;
        let v10 = p10[i] as f32;
        let v01 = p01[i] as f32;
        let v11 = p11[i] as f32;

        let val = (v00 * (1.0 - dx) * (1.0 - dy)) +
                  (v10 * dx * (1.0 - dy)) +
                  (v01 * (1.0 - dx) * dy) +
                  (v11 * dx * dy);
        out[i] = val.round() as u8;
    }

    Rgba(out)
}
