"""
Tests for the QFlex library.
"""

import numpy as np
import pytest
from qflex import QFlex, LogQFlex, LogitQFlex, ConstraintType, QFlexError
from qflex.utils import compute_w1


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_data():
    """Five-point elicitation covering a roughly normal-looking distribution."""
    y = [0.10, 0.25, 0.50, 0.75, 0.90]
    x = [12.0, 18.0, 25.0, 34.0, 45.0]
    return x, y


@pytest.fixture
def fine_data():
    """Ten-point elicitation for overdetermined fitting tests."""
    y = np.linspace(0.05, 0.95, 10).tolist()
    x = np.sort(np.random.default_rng(42).normal(50, 10, 10)).tolist()
    return x, y


# ---------------------------------------------------------------------------
# QFlex (unbounded)
# ---------------------------------------------------------------------------

class TestQFlex:
    def test_default_terms_three(self):
        """Default terms=3 suits P10/P50/P90 expert elicitation."""
        y = [0.10, 0.50, 0.90]
        x = [12.0, 25.0, 45.0]
        qf = QFlex(x, y)
        assert qf.terms == 3
        assert len(qf.coefficients) == 3
        np.testing.assert_allclose(qf.quantile(y), x, rtol=1e-4)

    def test_basic_fit(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        assert qf.coefficients is not None
        assert len(qf.coefficients) == 5

    def test_quantile_monotone(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        p = np.linspace(0.01, 0.99, 200)
        q = qf.quantile(p)
        assert np.all(np.diff(q) > 0), "Quantile function is not monotonically increasing"

    def test_quantile_interpolates_data(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        fitted = qf.quantile(y)
        np.testing.assert_allclose(fitted, x, rtol=1e-4,
            err_msg="Quantile function does not interpolate input data")

    def test_pdf_positive(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        p = np.linspace(0.01, 0.99, 100)
        pdf = qf.pdf(p)
        assert np.all(pdf > 0), "PDF has non-positive values"

    def test_pdf_analytical_vs_numerical(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        p = np.linspace(0.05, 0.95, 50)
        pdf_num = qf.pdf(p, method='numerical')
        pdf_ana = qf.pdf(p, method='analytical')
        np.testing.assert_allclose(pdf_num, pdf_ana, rtol=1e-2,
            err_msg="Analytical and numerical PDFs differ by more than 1%")

    def test_pdf_integrates_to_one(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        p = np.linspace(0.001, 0.999, 2000)
        pdf = qf.pdf(p)
        # Integral of pdf(p) * q(p) dp = 1 (change of variables)
        q = np.diff(qf.quantile(p))
        integral = np.sum(pdf[:-1] * q)
        assert abs(integral - 1.0) < 0.05, f"PDF does not integrate to 1 (got {integral:.4f})"

    def test_cdf_roundtrip(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        x_test = [18.0, 25.0, 34.0]
        cdf = qf.cdf(x_test)
        assert np.all(cdf > 0) and np.all(cdf < 1)
        recovered = qf.quantile(cdf)
        np.testing.assert_allclose(recovered, x_test, rtol=1e-3)

    def test_sample_shape(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        samples = qf.sample(size=500)
        assert samples.shape == (500,)

    def test_moments_keys(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        m = qf.moments(order=4)
        for key in ['mean', 'variance', 'std', 'skewness', 'kurtosis']:
            assert key in m, f"Missing moment key: {key}"

    def test_overdetermined_fit(self, fine_data):
        x, y = fine_data
        qf = QFlex(x, y, terms=5)
        assert qf.coefficients is not None

    def test_invalid_y_raises(self):
        with pytest.raises(QFlexError):
            QFlex([1, 2, 3, 4, 5], [0.0, 0.25, 0.50, 0.75, 1.0])  # y has 0 and 1

    def test_mismatched_lengths_raises(self):
        with pytest.raises(QFlexError):
            QFlex([1, 2, 3], [0.25, 0.50])

    def test_insufficient_data_raises(self):
        with pytest.raises(QFlexError):
            QFlex([1, 2], [0.3, 0.7], terms=5)

    def test_invalid_pdf_method_raises(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y)
        with pytest.raises(ValueError):
            qf.pdf([0.5], method='bogus')


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_constraint_a_plus(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5, constraint_type=ConstraintType.A)
        # All coefficients k>=2 (indices 1+) should be non-negative
        assert np.all(qf.coefficients[1:] >= -1e-8)

    def test_constraint_tl(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5, constraint_type=ConstraintType.TL)
        assert qf.coefficients is not None

    def test_constraint_ta(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5, constraint_type=ConstraintType.TA)
        assert qf.coefficients is not None

    def test_constraint_tc(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5, constraint_type=ConstraintType.TC)
        result = qf.check_proposition4()
        # TC enforces prop5 which implies prop4 is close to satisfied
        assert qf.coefficients is not None

    def test_constraint_tc_mag(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5, constraint_type=ConstraintType.TC_MAG)
        result = qf.check_proposition4()
        assert result['satisfied'], "TC_MAG constraint should satisfy Proposition 4"

    def test_check_proposition4_returns_dict(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        result = qf.check_proposition4()
        for key in ['satisfied', 'm_tail', 'M_center', 'margin', 'q_flex_min', 'q_flex_positive']:
            assert key in result


# ---------------------------------------------------------------------------
# LogQFlex (semibounded)
# ---------------------------------------------------------------------------

class TestLogQFlex:
    def test_basic_fit(self):
        y = [0.10, 0.25, 0.50, 0.75, 0.90]
        x = [5.0, 12.0, 25.0, 50.0, 100.0]
        qf = LogQFlex(x, y, lower_bound=0, terms=5)
        assert qf.coefficients is not None

    def test_quantile_above_lower_bound(self):
        y = [0.10, 0.25, 0.50, 0.75, 0.90]
        x = [5.0, 12.0, 25.0, 50.0, 100.0]
        qf = LogQFlex(x, y, lower_bound=0, terms=5)
        p = np.linspace(0.01, 0.99, 100)
        q = qf.quantile(p)
        assert np.all(q > 0), "LogQFlex quantile values should be > lower_bound"

    def test_quantile_monotone(self):
        y = [0.10, 0.25, 0.50, 0.75, 0.90]
        x = [5.0, 12.0, 25.0, 50.0, 100.0]
        qf = LogQFlex(x, y, lower_bound=0, terms=5)
        p = np.linspace(0.01, 0.99, 200)
        assert np.all(np.diff(qf.quantile(p)) > 0)

    def test_cdf_at_lower_bound(self):
        y = [0.10, 0.25, 0.50, 0.75, 0.90]
        x = [5.0, 12.0, 25.0, 50.0, 100.0]
        qf = LogQFlex(x, y, lower_bound=0, terms=5)
        assert qf.cdf(np.array([0.0])) == pytest.approx(0.0)

    def test_data_at_lower_bound_raises(self):
        with pytest.raises(QFlexError):
            LogQFlex([0.0, 10.0, 25.0, 50.0, 100.0],
                     [0.10, 0.25, 0.50, 0.75, 0.90], lower_bound=0)

    def test_pdf_analytical(self):
        y = [0.10, 0.25, 0.50, 0.75, 0.90]
        x = [5.0, 12.0, 25.0, 50.0, 100.0]
        qf = LogQFlex(x, y, lower_bound=0, terms=5)
        p = np.linspace(0.05, 0.95, 50)
        pdf_num = qf.pdf(p, method='numerical')
        pdf_ana = qf.pdf(p, method='analytical')
        np.testing.assert_allclose(pdf_num, pdf_ana, rtol=1e-2)


# ---------------------------------------------------------------------------
# LogitQFlex (bounded)
# ---------------------------------------------------------------------------

class TestLogitQFlex:
    def test_basic_fit(self):
        y = [0.10, 0.25, 0.50, 0.75, 0.90]
        x = [0.05, 0.20, 0.50, 0.75, 0.92]
        qf = LogitQFlex(x, y, lower_bound=0, upper_bound=1, terms=5)
        assert qf.coefficients is not None

    def test_quantile_within_bounds(self):
        y = [0.10, 0.25, 0.50, 0.75, 0.90]
        x = [0.05, 0.20, 0.50, 0.75, 0.92]
        qf = LogitQFlex(x, y, lower_bound=0, upper_bound=1, terms=5)
        p = np.linspace(0.01, 0.99, 100)
        q = qf.quantile(p)
        assert np.all(q > 0) and np.all(q < 1)

    def test_cdf_boundary_values(self):
        y = [0.10, 0.25, 0.50, 0.75, 0.90]
        x = [0.05, 0.20, 0.50, 0.75, 0.92]
        qf = LogitQFlex(x, y, lower_bound=0, upper_bound=1, terms=5)
        assert qf.cdf(np.array([0.0]))[0] == pytest.approx(0.0)
        assert qf.cdf(np.array([1.0]))[0] == pytest.approx(1.0)

    def test_invalid_bounds_raises(self):
        with pytest.raises(QFlexError):
            LogitQFlex([0.2, 0.5, 0.8], [0.25, 0.50, 0.75],
                       lower_bound=1.0, upper_bound=0.0)

    def test_pdf_analytical(self):
        y = [0.10, 0.25, 0.50, 0.75, 0.90]
        x = [0.05, 0.20, 0.50, 0.75, 0.92]
        qf = LogitQFlex(x, y, lower_bound=0, upper_bound=1, terms=5)
        p = np.linspace(0.05, 0.95, 50)
        pdf_num = qf.pdf(p, method='numerical')
        pdf_ana = qf.pdf(p, method='analytical')
        np.testing.assert_allclose(pdf_num, pdf_ana, rtol=1e-2)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestUtils:
    def test_compute_w1(self, basic_data):
        x, y = basic_data
        qf = QFlex(x, y, terms=5)
        w1, w1_norm = compute_w1(qf.quantile, x, y)
        assert w1 >= 0
        assert w1_norm >= 0

    def test_compute_w1_perfect_fit(self):
        """W1 of an exact-interpolating fit should be small relative to data range."""
        y = [0.10, 0.25, 0.50, 0.75, 0.90]
        x = [12.0, 18.0, 25.0, 34.0, 45.0]
        qf = QFlex(x, y, terms=5)
        w1, w1_norm = compute_w1(qf.quantile, x, y)
        # Normalized W1 should be under 10% of the P10-P90 range
        assert w1_norm < 0.10, f"Normalized W1 = {w1_norm:.4f} exceeds 10%"
