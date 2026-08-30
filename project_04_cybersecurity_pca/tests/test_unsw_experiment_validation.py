from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

import cyber_pca.unsw_experiment as experiment
from cyber_pca.unsw_preprocessing import (
    UNSWPreprocessor,
    UNSWStandardizedDataSplits,
)


FEATURE_NAMES = (
    "feature_a",
    "feature_b",
    "feature_c",
)


def _frame(
    values: object,
    identifiers: tuple[object, ...],
) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(
            values,
            dtype=np.float64,
        ),
        columns=FEATURE_NAMES,
        index=pd.Index(
            identifiers,
            name="flow_id",
        ),
    )


def _valid_splits() -> (
    UNSWStandardizedDataSplits
):
    normal_fit = _frame(
        (
            (-1.0, -1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
        ),
        (
            "fit:1",
            "fit:2",
            "fit:3",
            "fit:4",
        ),
    )

    normal_calibration = _frame(
        (
            (0.5, 0.2, -0.3),
            (-0.5, -0.2, 0.3),
        ),
        (
            "calibration:1",
            "calibration:2",
        ),
    )

    test = _frame(
        (
            (2.0, 0.0, 1.0),
            (0.0, 2.0, -1.0),
        ),
        (
            "test:1",
            "test:2",
        ),
    )

    scaler = StandardScaler().fit(
        normal_fit.to_numpy(
            dtype=np.float64,
        )
    )

    preprocessor = UNSWPreprocessor(
        encoder=OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        ),
        scaler=scaler,
        feature_names=FEATURE_NAMES,
    )

    return UNSWStandardizedDataSplits(
        normal_fit=normal_fit,
        normal_calibration=(
            normal_calibration
        ),
        test=test,
        preprocessor=preprocessor,
    )


def _replace_fit_frame(
    splits: UNSWStandardizedDataSplits,
    frame: object,
) -> UNSWStandardizedDataSplits:
    return replace(
        splits,
        normal_fit=frame,
    )


def test_readonly_copy_is_float64_and_immutable() -> None:
    source = np.array(
        [1, 2, 3],
        dtype=np.int64,
    )

    copied = (
        experiment._readonly_float64_copy(
            source
        )
    )

    assert copied.dtype == np.float64
    assert copied.flags.writeable is False
    assert not np.shares_memory(
        source,
        copied,
    )

    with pytest.raises(
        ValueError,
        match="read-only",
    ):
        copied[0] = 99.0


def test_rejects_non_unsw_split_container() -> None:
    with pytest.raises(
        TypeError,
        match="UNSWStandardizedDataSplits",
    ):
        experiment.fit_unsw_normal_pca(
            object()
        )


def test_rejects_invalid_preprocessor() -> None:
    splits = _valid_splits()

    invalid = replace(
        splits,
        preprocessor=object(),
    )

    with pytest.raises(
        TypeError,
        match="UNSWPreprocessor",
    ):
        experiment.fit_unsw_normal_pca(
            invalid
        )


def test_rejects_invalid_scaler_type() -> None:
    splits = _valid_splits()

    preprocessor = replace(
        splits.preprocessor,
        scaler=object(),
    )

    invalid = replace(
        splits,
        preprocessor=preprocessor,
    )

    with pytest.raises(
        TypeError,
        match="StandardScaler",
    ):
        experiment.fit_unsw_normal_pca(
            invalid
        )


@pytest.mark.parametrize(
    (
        "feature_names",
        "exception_type",
        "message",
    ),
    (
        (
            (),
            ValueError,
            "must not be empty",
        ),
        (
            (
                "feature_a",
                "",
                "feature_c",
            ),
            TypeError,
            "nonempty strings",
        ),
        (
            (
                "feature_a",
                "feature_a",
                "feature_c",
            ),
            ValueError,
            "must be unique",
        ),
    ),
)
def test_rejects_invalid_feature_names(
    feature_names: tuple[object, ...],
    exception_type: type[Exception],
    message: str,
) -> None:
    splits = _valid_splits()

    preprocessor = replace(
        splits.preprocessor,
        feature_names=feature_names,
    )

    invalid = replace(
        splits,
        preprocessor=preprocessor,
    )

    with pytest.raises(
        exception_type,
        match=message,
    ):
        experiment.fit_unsw_normal_pca(
            invalid
        )


@pytest.mark.parametrize(
    (
        "case",
        "exception_type",
        "message",
    ),
    (
        (
            "not_dataframe",
            TypeError,
            "pandas DataFrame",
        ),
        (
            "empty",
            ValueError,
            "must not be empty",
        ),
        (
            "wrong_columns",
            ValueError,
            "exactly match feature_names",
        ),
        (
            "unnamed_index",
            ValueError,
            "index must be named flow_id",
        ),
        (
            "missing_id",
            ValueError,
            "missing flow IDs",
        ),
        (
            "duplicate_id",
            ValueError,
            "duplicate flow IDs",
        ),
        (
            "nonnumeric",
            TypeError,
            "numeric and nonboolean",
        ),
        (
            "boolean",
            TypeError,
            "numeric and nonboolean",
        ),
        (
            "nonfinite",
            ValueError,
            "nonfinite model values",
        ),
    ),
)
def test_rejects_invalid_model_frame(
    case: str,
    exception_type: type[Exception],
    message: str,
) -> None:
    splits = _valid_splits()
    frame: object

    if case == "not_dataframe":
        frame = np.zeros((4, 3))
    elif case == "empty":
        frame = pd.DataFrame(
            columns=FEATURE_NAMES,
            index=pd.Index(
                [],
                name="flow_id",
            ),
            dtype=np.float64,
        )
    else:
        frame = splits.normal_fit.copy(
            deep=True
        )

        if case == "wrong_columns":
            frame.columns = (
                "wrong_a",
                "wrong_b",
                "wrong_c",
            )
        elif case == "unnamed_index":
            frame.index = frame.index.rename(
                None
            )
        elif case == "missing_id":
            identifiers = list(frame.index)
            identifiers[0] = None
            frame.index = pd.Index(
                identifiers,
                name="flow_id",
            )
        elif case == "duplicate_id":
            identifiers = list(frame.index)
            identifiers[1] = identifiers[0]
            frame.index = pd.Index(
                identifiers,
                name="flow_id",
            )
        elif case == "nonnumeric":
            frame["feature_a"] = (
                frame["feature_a"]
                .astype(object)
            )
            frame.loc[
                frame.index[0],
                "feature_a",
            ] = "invalid"
        elif case == "boolean":
            frame["feature_a"] = (
                frame["feature_a"] > 0.0
            )
        elif case == "nonfinite":
            frame.loc[
                frame.index[0],
                "feature_a",
            ] = np.inf
        else:
            raise AssertionError(
                f"Unsupported case: {case}"
            )

    invalid = _replace_fit_frame(
        splits,
        frame,
    )

    with pytest.raises(
        exception_type,
        match=message,
    ):
        experiment.fit_unsw_normal_pca(
            invalid
        )


def test_rejects_single_normal_fit_observation() -> None:
    splits = _valid_splits()

    frame = (
        splits.normal_fit
        .iloc[[0]]
        .copy(deep=True)
    )

    invalid = _replace_fit_frame(
        splits,
        frame,
    )

    with pytest.raises(
        ValueError,
        match="at least two observations",
    ):
        experiment.fit_unsw_normal_pca(
            invalid
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        (
            "fit_calibration",
            "Fitting and calibration",
        ),
        (
            "fit_test",
            "Fitting and test",
        ),
        (
            "calibration_test",
            "Calibration and test",
        ),
    ),
)
def test_rejects_overlapping_flow_ids(
    case: str,
    message: str,
) -> None:
    splits = _valid_splits()

    normal_calibration = (
        splits.normal_calibration.copy(
            deep=True
        )
    )

    test = splits.test.copy(
        deep=True
    )

    if case == "fit_calibration":
        identifiers = list(
            normal_calibration.index
        )
        identifiers[0] = (
            splits.normal_fit.index[0]
        )
        normal_calibration.index = pd.Index(
            identifiers,
            name="flow_id",
        )
    elif case == "fit_test":
        identifiers = list(test.index)
        identifiers[0] = (
            splits.normal_fit.index[0]
        )
        test.index = pd.Index(
            identifiers,
            name="flow_id",
        )
    elif case == "calibration_test":
        identifiers = list(test.index)
        identifiers[0] = (
            normal_calibration.index[0]
        )
        test.index = pd.Index(
            identifiers,
            name="flow_id",
        )
    else:
        raise AssertionError(
            f"Unsupported case: {case}"
        )

    invalid = replace(
        splits,
        normal_calibration=(
            normal_calibration
        ),
        test=test,
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        experiment.fit_unsw_normal_pca(
            invalid
        )


def test_rejects_unfitted_scaler() -> None:
    splits = _valid_splits()

    preprocessor = replace(
        splits.preprocessor,
        scaler=StandardScaler(),
    )

    invalid = replace(
        splits,
        preprocessor=preprocessor,
    )

    with pytest.raises(
        ValueError,
        match="must be fitted",
    ):
        experiment.fit_unsw_normal_pca(
            invalid
        )


def test_rejects_scaler_feature_count_mismatch() -> None:
    splits = _valid_splits()

    scaler = StandardScaler().fit(
        np.array(
            (
                (0.0, 1.0),
                (1.0, 0.0),
            ),
            dtype=np.float64,
        )
    )

    preprocessor = replace(
        splits.preprocessor,
        scaler=scaler,
    )

    invalid = replace(
        splits,
        preprocessor=preprocessor,
    )

    with pytest.raises(
        ValueError,
        match="feature count",
    ):
        experiment.fit_unsw_normal_pca(
            invalid
        )


@pytest.mark.parametrize(
    (
        "target",
        "exception_type",
        "message",
    ),
    (
        (
            True,
            TypeError,
            "must be numeric",
        ),
        (
            "0.95",
            TypeError,
            "must be numeric",
        ),
        (
            np.nan,
            ValueError,
            "must be finite",
        ),
        (
            0.0,
            ValueError,
            r"\(0, 1\]",
        ),
        (
            1.01,
            ValueError,
            r"\(0, 1\]",
        ),
    ),
)
def test_rejects_invalid_variance_target(
    target: object,
    exception_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(
        exception_type,
        match=message,
    ):
        experiment.fit_unsw_normal_pca(
            _valid_splits(),
            explained_variance_target=target,
        )


def test_rejects_inconsistent_refit_eigenvalues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_class = (
        experiment.ManualPCA
    )

    state = {"fits": 0}

    class InconsistentManualPCA(
        original_class
    ):
        def fit(
            self,
            values: object,
        ) -> InconsistentManualPCA:
            result = super().fit(values)
            state["fits"] += 1

            if state["fits"] == 2:
                changed = (
                    self
                    .all_explained_variance_
                    .copy()
                )
                changed[0] += 1.0
                self.all_explained_variance_ = (
                    changed
                )

            return result

    monkeypatch.setattr(
        experiment,
        "ManualPCA",
        InconsistentManualPCA,
    )

    with pytest.raises(
        RuntimeError,
        match="inconsistent eigenvalues",
    ):
        experiment.fit_unsw_normal_pca(
            _valid_splits()
        )


def test_reconstruction_rejects_invalid_fit_result() -> None:
    with pytest.raises(
        TypeError,
        match="PCAFitResult",
    ):
        experiment.compute_unsw_reconstruction_errors(
            _valid_splits(),
            object(),
        )


def test_reconstruction_rejects_invalid_model() -> None:
    splits = _valid_splits()

    fit_result = (
        experiment.fit_unsw_normal_pca(
            splits
        )
    )

    invalid = replace(
        fit_result,
        model=object(),
    )

    with pytest.raises(
        TypeError,
        match="ManualPCA",
    ):
        experiment.compute_unsw_reconstruction_errors(
            splits,
            invalid,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        (
            "wrong_shape",
            "unexpected shape",
        ),
        (
            "nonfinite",
            "must be finite",
        ),
        (
            "negative",
            "must be nonnegative",
        ),
    ),
)
def test_rejects_invalid_reconstruction_output(
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splits = _valid_splits()

    fit_result = (
        experiment.fit_unsw_normal_pca(
            splits
        )
    )

    def invalid_reconstruction_error(
        self: object,
        matrix: object,
    ) -> np.ndarray:
        rows = np.asarray(matrix).shape[0]

        if case == "wrong_shape":
            return np.zeros(
                rows + 1,
                dtype=np.float64,
            )

        values = np.zeros(
            rows,
            dtype=np.float64,
        )

        if case == "nonfinite":
            values[0] = np.nan
        elif case == "negative":
            values[0] = -1.0
        else:
            raise AssertionError(
                f"Unsupported case: {case}"
            )

        return values

    monkeypatch.setattr(
        experiment.ManualPCA,
        "reconstruction_error",
        invalid_reconstruction_error,
    )

    with pytest.raises(
        RuntimeError,
        match=message,
    ):
        experiment.compute_unsw_reconstruction_errors(
            splits,
            fit_result,
        )
