//! Jacobi-preconditioned conjugate gradient on a compact local CSR system, with `K`
//! right-hand sides solved simultaneously (they share the matrix, so the SpMV reads the
//! matrix once for all of them). All hot data is `f32`; reductions accumulate in `f64`
//! over fixed-size row chunks and are summed in chunk order, so results are bit-identical
//! whether the kernels run sequentially or on rayon.

use rayon::prelude::*;

/// Rows per reduction chunk (fixed => deterministic reduction order).
pub(crate) const CH: usize = 2048;

/// Compact SPD system `A = D - Off` for a set of unknown nodes. Off-diagonals only couple
/// nodes inside the set; couplings to Dirichlet nodes (anchors, or frozen ring nodes in
/// local mode) only contribute to the diagonal.
#[derive(Clone, Default, Debug)]
pub(crate) struct LocalSys {
    /// Global node id of each local row.
    pub nodes: Vec<u32>,
    pub row_ptr: Vec<u32>,
    pub col: Vec<u32>,
    pub val: Vec<f32>,
    pub diag: Vec<f32>,
    pub inv_diag: Vec<f32>,
}

impl LocalSys {
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.row_ptr.clear();
        self.col.clear();
        self.val.clear();
        self.diag.clear();
        self.inv_diag.clear();
    }
    pub fn n(&self) -> usize {
        self.nodes.len()
    }
}

/// Flat (stride `K`) work vectors, reused between calls to avoid per-tick allocation.
#[derive(Clone, Default, Debug)]
pub(crate) struct Work {
    pub b: Vec<f32>,
    pub x: Vec<f32>,
    pub r: Vec<f32>,
    pub z: Vec<f32>,
    pub p: Vec<f32>,
    pub q: Vec<f32>,
}

#[inline]
fn spmv_chunk<const K: usize>(sys: &LocalSys, p: &[f32], start: usize, qc: &mut [f32]) -> [f64; K] {
    let (p, _) = p.as_chunks::<K>();
    let (qc, _) = qc.as_chunks_mut::<K>();
    let n = qc.len();
    let diag = &sys.diag[start..start + n];
    let rp = &sys.row_ptr[start..start + n + 1];
    let mut pq = [0f64; K];
    for li in 0..n {
        let i = start + li;
        let pi = p[i];
        let d = diag[li];
        let mut acc = [0f32; K];
        for c in 0..K {
            acc[c] = d * pi[c];
        }
        let (s, e) = (rp[li] as usize, rp[li + 1] as usize);
        for (&j, &v) in sys.col[s..e].iter().zip(&sys.val[s..e]) {
            let pj = p[j as usize];
            for c in 0..K {
                acc[c] -= v * pj[c];
            }
        }
        qc[li] = acc;
        for c in 0..K {
            pq[c] += (acc[c] * pi[c]) as f64;
        }
    }
    pq
}

/// x += a p; r -= a q; z = r / diag. Returns (r.z, max|r|) per component.
#[inline]
#[allow(clippy::too_many_arguments)]
fn update_chunk<const K: usize>(
    alpha: [f32; K],
    pc: &[f32],
    qc: &[f32],
    inv_d: &[f32],
    xc: &mut [f32],
    rc: &mut [f32],
    zc: &mut [f32],
) -> ([f64; K], [f32; K]) {
    let mut rz = [0f64; K];
    let mut rmax = [0f32; K];
    for li in 0..inv_d.len() {
        let id = inv_d[li];
        for c in 0..K {
            let k = li * K + c;
            xc[k] += alpha[c] * pc[k];
            let r = rc[k] - alpha[c] * qc[k];
            rc[k] = r;
            let z = r * id;
            zc[k] = z;
            rz[c] += (r * z) as f64;
            rmax[c] = rmax[c].max(r.abs());
        }
    }
    (rz, rmax)
}

fn add<const K: usize>(a: ([f64; K], [f32; K]), b: ([f64; K], [f32; K])) -> ([f64; K], [f32; K]) {
    let mut o = a;
    for c in 0..K {
        o.0[c] += b.0[c];
        o.1[c] = o.1[c].max(b.1[c]);
    }
    o
}

/// Solve `A x = b` (b already in `w.b`, length n*K) from x = 0 for each of the K columns,
/// stopping when every column satisfies `max_i |r_i| <= thr[c]` or after `max_iters`
/// SpMVs. Leaves the final (recursive) residual in `w.r`. Returns (iterations, converged).
pub(crate) fn pcg<const K: usize>(
    sys: &LocalSys,
    w: &mut Work,
    thr: [f32; K],
    max_iters: usize,
    par: bool,
) -> (usize, bool) {
    let n = sys.n();
    let nk = n * K;
    debug_assert_eq!(w.b.len(), nk);
    for v in [&mut w.x, &mut w.r, &mut w.z, &mut w.p, &mut w.q] {
        v.clear();
        v.resize(nk, 0.0);
    }
    if n == 0 {
        return (0, true);
    }
    // r = b, z = M^-1 r, p = z
    w.r.copy_from_slice(&w.b);
    let (mut rz, rmax) = {
        let zero = [0f32; K];
        let pz = std::mem::take(&mut w.p);
        let qz = std::mem::take(&mut w.q);
        let res = w
            .x
            .chunks_mut(CH * K)
            .zip(w.r.chunks_mut(CH * K))
            .zip(w.z.chunks_mut(CH * K))
            .zip(pz.chunks(CH * K).zip(qz.chunks(CH * K)))
            .zip(sys.inv_diag.chunks(CH))
            .map(|((((xc, rc), zc), (pc, qc)), id)| update_chunk::<K>(zero, pc, qc, id, xc, rc, zc))
            .fold(([0f64; K], [0f32; K]), add::<K>);
        w.p = pz;
        w.q = qz;
        res
    };
    w.p.copy_from_slice(&w.z);
    let mut done = [false; K];
    for c in 0..K {
        done[c] = rmax[c] <= thr[c];
    }
    let mut iters = 0;
    while !done.iter().all(|&d| d) && iters < max_iters {
        // q = A p, pq = p.q
        let pq: [f64; K] = if par {
            let p = &w.p;
            let parts: Vec<[f64; K]> = w
                .q
                .par_chunks_mut(CH * K)
                .enumerate()
                .map(|(ci, qc)| spmv_chunk::<K>(sys, p, ci * CH, qc))
                .collect();
            sum_parts(&parts)
        } else {
            let p = &w.p;
            let mut acc = [0f64; K];
            for (ci, qc) in w.q.chunks_mut(CH * K).enumerate() {
                let part = spmv_chunk::<K>(sys, p, ci * CH, qc);
                for c in 0..K {
                    acc[c] += part[c];
                }
            }
            acc
        };
        let mut alpha = [0f32; K];
        for c in 0..K {
            if !done[c] && pq[c] > 0.0 {
                alpha[c] = (rz[c] / pq[c]) as f32;
            }
        }
        let (rz_new, rmax) = if par {
            let parts: Vec<([f64; K], [f32; K])> = w
                .x
                .par_chunks_mut(CH * K)
                .zip(w.r.par_chunks_mut(CH * K))
                .zip(w.z.par_chunks_mut(CH * K))
                .zip(w.p.par_chunks(CH * K).zip(w.q.par_chunks(CH * K)))
                .zip(sys.inv_diag.par_chunks(CH))
                .map(|((((xc, rc), zc), (pc, qc)), id)| update_chunk::<K>(alpha, pc, qc, id, xc, rc, zc))
                .collect();
            parts.into_iter().fold(([0f64; K], [0f32; K]), add::<K>)
        } else {
            w.x.chunks_mut(CH * K)
                .zip(w.r.chunks_mut(CH * K))
                .zip(w.z.chunks_mut(CH * K))
                .zip(w.p.chunks(CH * K).zip(w.q.chunks(CH * K)))
                .zip(sys.inv_diag.chunks(CH))
                .map(|((((xc, rc), zc), (pc, qc)), id)| update_chunk::<K>(alpha, pc, qc, id, xc, rc, zc))
                .fold(([0f64; K], [0f32; K]), add::<K>)
        };
        iters += 1;
        let mut beta = [0f32; K];
        for c in 0..K {
            if !done[c] && rmax[c] <= thr[c] {
                done[c] = true;
            }
            if !done[c] && rz[c] > 0.0 {
                beta[c] = (rz_new[c] / rz[c]) as f32;
            }
            if !done[c] {
                rz[c] = rz_new[c];
            }
        }
        if done.iter().all(|&d| d) {
            break;
        }
        // p = z + beta p
        let upd = |pc: &mut [f32], zc: &[f32]| {
            for (pp, zz) in pc.as_chunks_mut::<K>().0.iter_mut().zip(zc.as_chunks::<K>().0) {
                for c in 0..K {
                    pp[c] = zz[c] + beta[c] * pp[c];
                }
            }
        };
        if par {
            w.p.par_chunks_mut(CH * K).zip(w.z.par_chunks(CH * K)).for_each(|(pc, zc)| upd(pc, zc));
        } else {
            upd(&mut w.p, &w.z);
        }
    }
    (iters, done.iter().all(|&d| d))
}

fn sum_parts<const K: usize>(parts: &[[f64; K]]) -> [f64; K] {
    let mut acc = [0f64; K];
    for p in parts {
        for c in 0..K {
            acc[c] += p[c];
        }
    }
    acc
}
