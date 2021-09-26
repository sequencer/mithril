#[cfg(all(
any(target_arch = "x86", target_arch = "x86_64"),
target_feature = "sse2"
))]
use std::arch::x86_64::{
    __m128d, __m128i, _mm_add_pd, _mm_aesdec_si128, _mm_aesenc_si128, _mm_and_pd, _mm_cmpeq_epi32,
    _mm_cmpeq_pd, _mm_cvtepi32_pd, _mm_div_pd, _mm_extract_epi64, _mm_movemask_epi8,
    _mm_movemask_pd, _mm_mul_pd, _mm_or_pd, _mm_set_epi32, _mm_set_epi64x, _mm_set_pd,
    _mm_shuffle_pd, _mm_sqrt_pd, _mm_store_sd, _mm_storeh_pd, _mm_sub_pd, _mm_xor_pd,
};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{float64x2_t, int32x4_t, vld1q_s32, vceqq_f64, vceqq_s32, vaesdq_u8, vaeseq_u8, vaddq_f64, vsubq_f64, veorq_u32, vandq_u32, vorrq_u32, vmulq_f64};
use std::mem::transmute;
use std::convert::TryInto;
use std::fmt;

#[allow(nonstandard_style)]
#[derive(Copy, Clone)]
#[cfg(all(
any(target_arch = "x86", target_arch = "x86_64"),
target_feature = "sse2"
))]
pub struct m128i(pub __m128i);

#[allow(nonstandard_style)]
#[derive(Copy, Clone)]
#[cfg(target_arch = "aarch64")]
pub struct m128i(pub int32x4_t);

impl m128i {
    pub fn zero() -> m128i {
        m128i::from_i32(0, 0, 0, 0)
    }
    pub fn from_u8(bytes: &[u8]) -> m128i {
        debug_assert_eq!(bytes.len(), 16);

        let u0 = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let u1 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());

        m128i::from_u64(u1, u0)
    }
    pub fn from_i32(i3: i32, i2: i32, i1: i32, i0: i32) -> m128i {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128i(_mm_set_epi32(i3, i2, i1, i0)) }
        #[cfg(target_arch = "aarch64")]
            unsafe {
            let ptr = [i2, i3, i0, i1];
            return m128i(vld1q_s32(&ptr as *const i32));
        }
    }
    pub fn from_u64(u1: u64, u0: u64) -> m128i {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128i(_mm_set_epi64x(u1 as i64, u0 as i64)) }
        #[cfg(target_arch = "aarch64")]
            unsafe {
            let ptr = [u1, u0];
            return m128i(transmute(ptr));
        }
    }
    pub fn aesdec(&self, key: m128i) -> m128i {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128i(_mm_aesdec_si128(self.0, key.0)) }
        #[cfg(target_arch = "aarch64")]
            unsafe { m128i(transmute(vaesdq_u8(transmute(self.0), transmute(key)))) }
    }
    pub fn aesenc(&self, key: m128i) -> m128i {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128i(_mm_aesenc_si128(self.0, key.0)) }
        #[cfg(target_arch = "aarch64")]
            unsafe {
            m128i(transmute(vaeseq_u8(transmute(self.0), transmute(key))))
        }
    }
    pub fn as_i64(&self) -> (i64, i64) {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe {
            let p1 = _mm_extract_epi64(self.0, 1);
            let p2 = _mm_extract_epi64(self.0, 0);
            (p1, p2)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let (p1, p2) = transmute(self.0);
            (p1, p2)
        }
    }

    //_mm_cvtepi32_pd
    pub fn lower_to_m128d(&self) -> m128d {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_cvtepi32_pd(self.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let bits: [u32; 4] = transmute(self.0);
            let high: f64 = bits[3] as f64;
            let low: f64 = bits[2] as f64;
            m128d::from_f64(high, low)
        }
    }

    pub fn as_m128d(&self) -> m128d {
        let (i1, i0) = self.as_i64();
        m128d::from_u64(i1 as u64, i0 as u64)
    }
}

impl PartialEq for m128i {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe {
            let test = _mm_cmpeq_epi32(self.0, other.0);
            _mm_movemask_epi8(test) == 0xffff
        }
        #[cfg(target_arch = "aarch64")]
            unsafe {
            let mask = vceqq_s32(self.0, other.0);
            let mask: u128 = transmute(mask);
            mask == 0xffffffffffffffffffffffffffffffff
        }
    }
}

impl Eq for m128i {}

fn format_m128i(m: &m128i, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let (low, high) = m.as_i64();
    f.write_fmt(format_args!("({:x},{:x})", high, low))
}

impl fmt::LowerHex for m128i {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_m128i(self, f)
    }
}

impl fmt::Debug for m128i {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_m128i(self, f)
    }
}

//==== m128d

#[cfg(all(
any(target_arch = "x86", target_arch = "x86_64"),
target_feature = "sse2"
))]
#[allow(nonstandard_style)]
#[derive(Copy, Clone)]
pub struct m128d(pub __m128d);

#[cfg(target_arch = "aarch64")]
#[allow(nonstandard_style)]
#[derive(Copy, Clone)]
pub struct m128d(pub float64x2_t);

impl m128d {
    pub fn zero() -> m128d {
        m128d::from_f64(0.0, 0.0)
    }
    pub fn from_u64(h: u64, l: u64) -> m128d {
        m128d::from_f64(f64::from_bits(h), f64::from_bits(l))
    }
    pub fn from_f64(h: f64, l: f64) -> m128d {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_set_pd(h, l)) }
        #[cfg(target_arch = "aarch64")]
            unsafe {
            let ptr = [h, l];
            return m128d(transmute(ptr));
        }
    }
    pub fn as_f64(&self) -> (f64, f64) {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
        {
            let mut f1: f64 = 0.0;
            let mut f2: f64 = 0.0;
            let f1_ptr: *mut f64 = &mut f1;
            let f2_ptr: *mut f64 = &mut f2;
            unsafe {
                _mm_storeh_pd(f1_ptr, self.0);
                _mm_store_sd(f2_ptr, self.0);
            }
            (f1, f2)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            return transmute(self.0);
        }
    }

    pub fn as_u64(&self) -> (u64, u64) {
        let (f1, f0) = self.as_f64();
        (f1.to_bits(), f0.to_bits())
    }

    //_mm_shuffle_pd(a, b, 1)
    pub fn shuffle_1(&self, other: &m128d) -> m128d {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_shuffle_pd(self.0, other.0, 1)) }
        #[cfg(target_arch = "aarch64")]
            {
                let r: float64x2_t;
                unsafe {
                    asm!(
                    "ext.16b v2, v1, v0, #8",
                    in("v0") self.0,
                    in("v1") other.0,
                    out("v2") r,
                    );
                }
                m128d(r)
            }
    }

    //_mm_sqrt_pd
    pub fn sqrt(&self) -> m128d {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_sqrt_pd(self.0)) }
        #[cfg(target_arch = "aarch64")]
            {
                let r: float64x2_t;
                unsafe {
                    asm!(
                    "fsqrt.2d v2, v0",
                    in("v0") self.0,
                    out("v2") r,
                    );
                }
                m128d(r)
            }
    }
}

impl PartialEq for m128d {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe {
            let test = _mm_cmpeq_pd(self.0, other.0);
            let mask = _mm_movemask_pd(test);
            mask == 0b11
        }
        #[cfg(target_arch = "aarch64")]
            unsafe {
            let mask = vceqq_f64(self.0, other.0);
            let mask: u128 = transmute(mask);
            mask == 0xffffffff_ffffffff_ffffffff_ffffffff
        }
    }
}

impl Eq for m128d {}

impl std::ops::Add for m128d {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_add_pd(self.0, other.0)) }
        #[cfg(target_arch = "aarch64")]
            unsafe { return m128d(vaddq_f64(self.0, other.0)); }
    }
}

impl std::ops::Sub for m128d {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_sub_pd(self.0, other.0)) }
        #[cfg(target_arch = "aarch64")]
            unsafe { return m128d(vsubq_f64(self.0, other.0)); }
    }
}

fn format_m128d(m: &m128d, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let (low, high) = m.as_f64();
    f.write_fmt(format_args!("({},{})", low, high))
}

impl fmt::LowerHex for m128d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (low, high) = self.as_f64();
        f.write_fmt(format_args!("({:x},{:x})", high.to_bits(), low.to_bits()))
    }
}

impl fmt::Debug for m128d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_m128d(self, f)
    }
}

impl std::ops::BitXor for m128d {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_xor_pd(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
            unsafe { return m128d(transmute(veorq_u32(transmute(self.0), transmute(rhs.0)))); }
    }
}

impl std::ops::BitAnd for m128d {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_and_pd(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
            unsafe { return m128d(transmute(vandq_u32(transmute(self.0), transmute(rhs.0)))); }
    }
}

impl std::ops::BitOr for m128d {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_or_pd(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
            unsafe { return m128d(transmute(vorrq_u32(transmute(self.0), transmute(rhs.0)))); }
    }
}

impl std::ops::Mul for m128d {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_mul_pd(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
            unsafe { return m128d(vmulq_f64(self.0, rhs.0)); }
    }
}

impl std::ops::Div for m128d {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
        ))]
            unsafe { m128d(_mm_div_pd(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
            {
                let lhs = self.0;
                let rhs = rhs.0;
                let r: float64x2_t;
                unsafe {
                    asm!(
                    "fdiv.2d v2, v0, v1",
                    in("v0") lhs,
                    in("v1") rhs,
                    out("v2") r,
                    );
                    m128d(r)
                }
            }
    }
}
