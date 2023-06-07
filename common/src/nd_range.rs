use nalgebra_glm::TVec;

#[derive(Copy, Clone)]
pub struct NDRange<const N: usize> {
    start: TVec<usize, N>,
    end: TVec<usize, N>,
}

impl<const N: usize> NDRange<N> {
    pub fn new(start: TVec<usize, N>, end: TVec<usize, N>) -> Self {
        assert!(start < end);
        Self { start, end }
    }

    pub fn of_size(size: TVec<usize, N>) -> Self {
        let start = TVec::zeros();
        Self::new(start, size)
    }
}

impl<const N: usize> IntoIterator for NDRange<N> {
    type Item = TVec<usize, N>;
    type IntoIter = NDRangeIter<N>;

    fn into_iter(self) -> Self::IntoIter {
        NDRangeIter {
            start: self.start,
            end: self.end,
            next: Some(self.start),
        }
    }
}

pub struct NDRangeIter<const N: usize> {
    start: TVec<usize, N>,
    end: TVec<usize, N>,
    next: Option<TVec<usize, N>>,
}

impl<const N: usize> Iterator for NDRangeIter<N> {
    type Item = TVec<usize, N>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.next.take()?;
        let mut next = result;

        for i in (0..N).rev() {
            let curr_comp = &mut next[i];
            *curr_comp += 1;

            if *curr_comp < self.end[i] {
                self.next = Some(next);
                break;
            } else {
                *curr_comp = self.start[i];
            }
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.end - self.start).add_scalar(1).fold(1, |s, v| s * v);
        (len, Some(len))
    }
}

impl<const N: usize> ExactSizeIterator for NDRangeIter<N> {}
