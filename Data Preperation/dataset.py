from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import gc
import random
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    EXP2_TRAIN_SOURCE,
    EXP3_TRAIN_SOURCE,
    EXP2_TEST_SOURCE,
    EXP3_TEST_SOURCE,
    MANIFEST_DIR,
    SCALER_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
    CLASS_NAMES,
    LABEL_TO_ID,
    LABEL_ALIASES,
    NUM_FEATURES,
    WINDOW_SIZE,
    STEP_SIZE,
    MAX_WINDOWS_PER_FILE,
    VALIDATION_FRACTION,
    RANDOM_SEED,
)


# ============================================================
# DATENSTRUKTUR
# ============================================================

@dataclass(frozen=True)
class FileRecord:
    """
    Repräsentiert eine einzelne CSV-Aufzeichnung.

    path:
        Vollständiger Pfad zur CSV-Datei.

    label:
        Vereinheitlichte Aktivitätsklasse.

    experiment:
        Herkunft der Datei: EXP2 oder EXP3.
    """

    path: Path
    label: str
    experiment: str


# ============================================================
# LABELS VEREINHEITLICHEN
# ============================================================

def canonicalize_label(value: object) -> str:
    """
    Vereinheitlicht unterschiedliche Schreibweisen der Klassen.

    Beispiele:

        sitdown   -> SIT-DOWN
        standup   -> STAND-UP
        sitting   -> SIT
        standing  -> STAND
        walking   -> WALK
    """

    label = str(value).strip().upper()

    label = label.replace("_", "-")
    label = label.replace(" ", "-")
    label = re.sub(r"-+", "-", label)

    compact_label = label.replace("-", "")

    # Direkte Schreibweise prüfen
    if label in LABEL_ALIASES:
        return LABEL_ALIASES[label]

    # Kompakte Schreibweise prüfen, z. B. SITDOWN
    if compact_label in LABEL_ALIASES:
        return LABEL_ALIASES[compact_label]

    return label


def detect_label_from_path(
    csv_path: Path,
) -> str | None:
    """
    Erkennt die Aktivität anhand der Ordnernamen.

    Beispiele:

        D:/Experiment2/train/sitdown/datei.csv
        D:/Experiment3/train/sit/datei.csv
    """

    for path_part in reversed(csv_path.parts):
        candidate = canonicalize_label(path_part)

        if candidate in LABEL_TO_ID:
            return candidate

    return None


def detect_label_from_csv(
    csv_path: Path,
) -> str | None:
    """
    Liest als Ausweichmöglichkeit das Label aus der CSV-Datei.

    Dabei wird nur die erste Zeile benötigt.
    """

    try:
        dataframe = pd.read_csv(
            csv_path,
            nrows=1,
        )

    except Exception:
        return None

    possible_label_columns = {
        "label",
        "activity",
        "class",
        "target",
        "action",
    }

    for column in dataframe.columns:
        normalized_name = str(column).strip().lower()

        if normalized_name not in possible_label_columns:
            continue

        if dataframe.empty:
            return None

        raw_label = dataframe[column].iloc[0]
        label = canonicalize_label(raw_label)

        if label in LABEL_TO_ID:
            return label

    return None


def determine_label(
    csv_path: Path,
) -> str:
    """
    Bestimmt das Label einer CSV-Datei.

    Reihenfolge:

    1. Klassenordner
    2. Label-Spalte der CSV-Datei
    """

    label = detect_label_from_path(csv_path)

    if label is not None:
        return label

    label = detect_label_from_csv(csv_path)

    if label is not None:
        return label

    raise ValueError(
        f"Klasse konnte nicht bestimmt werden: {csv_path}"
    )


# ============================================================
# DATEIEN FINDEN
# ============================================================

def discover_records(
    root: Path,
    experiment: str,
) -> list[FileRecord]:
    """
    Findet alle CSV-Dateien eines Datenordners.

    Für jede Datei werden Pfad, Label und Experiment gespeichert.
    """

    if not root.exists():
        raise FileNotFoundError(
            f"Datenordner wurde nicht gefunden: {root}"
        )

    csv_files = sorted(root.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Keine CSV-Dateien gefunden: {root}"
        )

    records: list[FileRecord] = []
    skipped_files = 0

    for csv_path in csv_files:
        try:
            label = determine_label(csv_path)

        except ValueError as error:
            skipped_files += 1
            print(f"Datei übersprungen: {error}")
            continue

        records.append(
            FileRecord(
                path=csv_path,
                label=label,
                experiment=experiment,
            )
        )

    print(
        f"{experiment}: {len(records)} gültige Dateien gefunden, "
        f"{skipped_files} übersprungen."
    )

    return records


# ============================================================
# FEATURE-SPALTEN EINLESEN
# ============================================================

def get_numeric_suffix(
    column: object,
) -> int | None:
    """
    Extrahiert die Zahl am Ende eines Spaltennamens.

    Unterstützte Beispiele:

        0
        255
        SC_0
        SC_255
        Subcarrier_1
        Subcarrier_256
    """

    column_name = str(column).strip()

    if column_name.isdigit():
        return int(column_name)

    match = re.search(
        r"(\d+)$",
        column_name,
    )

    if match is None:
        return None

    return int(match.group(1))


def load_features(
    csv_path: Path,
) -> np.ndarray:
    """
    Lädt die 256 CSI-Feature-Spalten einer CSV-Datei.

    Die Rückgabe hat die Form:

        (Anzahl Pakete, 256)

    Die unterschiedlichen Spaltennamen aus Experiment 2 und 3
    werden anhand ihrer numerischen Endung sortiert.
    """

    dataframe = pd.read_csv(csv_path)

    label_names = {
        "label",
        "activity",
        "class",
        "target",
        "action",
    }

    metadata_names = {
        "experiment",
        "packet",
        "packet-id",
        "packet_id",
        "timestamp",
        "time",
        "index",
        "filename",
        "file",
    }

    columns_to_remove = []

    for column in dataframe.columns:
        normalized_name = str(column).strip().lower()

        if normalized_name in label_names:
            columns_to_remove.append(column)

        elif normalized_name in metadata_names:
            columns_to_remove.append(column)

        elif normalized_name.startswith("unnamed"):
            columns_to_remove.append(column)

    feature_dataframe = dataframe.drop(
        columns=columns_to_remove,
        errors="ignore",
    )

    feature_columns = list(feature_dataframe.columns)

    # Zahlen aus den Spaltennamen bestimmen
    feature_indices = [
        get_numeric_suffix(column)
        for column in feature_columns
    ]

    # Spalten numerisch sortieren
    if all(index is not None for index in feature_indices):
        indexed_columns = list(
            zip(
                feature_indices,
                feature_columns,
            )
        )

        indexed_columns.sort(
            key=lambda item: item[0]
        )

        feature_columns = [
            column
            for _, column in indexed_columns
        ]

    feature_dataframe = feature_dataframe[
        feature_columns
    ]

    # Textwerte in numerische Werte umwandeln
    feature_dataframe = feature_dataframe.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Vollständig nicht numerische Spalten entfernen
    feature_dataframe = feature_dataframe.dropna(
        axis=1,
        how="all",
    )

    if feature_dataframe.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"{csv_path} besitzt "
            f"{feature_dataframe.shape[1]} Features "
            f"statt {NUM_FEATURES}."
        )

    values = feature_dataframe.to_numpy(
        dtype=np.float32
    )

    if values.ndim != 2:
        raise ValueError(
            f"Unerwartete Datenform in {csv_path}: "
            f"{values.shape}"
        )

    if values.shape[0] == 0:
        raise ValueError(
            f"Datei enthält keine Pakete: {csv_path}"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Datei enthält NaN- oder Inf-Werte: {csv_path}"
        )

    return values


# ============================================================
# TRAINING UND VALIDATION AUFTEILEN
# ============================================================

def split_training_records(
    records: list[FileRecord],
) -> tuple[list[FileRecord], list[FileRecord]]:
    """
    Teilt die vorhandenen Trainingsordner intern in:

        neues Training
        interne Validation

    Die Stratifizierung berücksichtigt sowohl das Experiment
    als auch die Aktivitätsklasse.

    Dadurch bleibt beispielsweise der Anteil von EXP2-SIT und
    EXP3-SIT in beiden Splits möglichst ähnlich.
    """

    if not records:
        raise ValueError(
            "Keine Trainingsdateien für die Aufteilung vorhanden."
        )

    strata = [
        f"{record.experiment}|{record.label}"
        for record in records
    ]

    train_records, validation_records = train_test_split(
        records,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=strata,
    )

    return (
        list(train_records),
        list(validation_records),
    )


# ============================================================
# MANIFESTE SPEICHERN
# ============================================================

def save_manifest(
    records: list[FileRecord],
    filename: str,
) -> None:
    """
    Speichert, welche CSV-Dateien zu welchem Split gehören.

    Dadurch ist später nachvollziehbar, welche Datei für
    Training, Validation oder Test verwendet wurde.
    """

    rows = []

    for record in records:
        rows.append({
            "path": str(record.path),
            "experiment": record.experiment,
            "label": record.label,
            "label_id": LABEL_TO_ID[record.label],
        })

    manifest = pd.DataFrame(rows)

    output_path = MANIFEST_DIR / filename

    manifest.to_csv(
        output_path,
        index=False,
    )

    print(f"Manifest gespeichert: {output_path}")


# ============================================================
# DATEIVERTEILUNG AUSGEBEN
# ============================================================

def print_record_distribution(
    name: str,
    records: list[FileRecord],
) -> None:
    """
    Zeigt die Anzahl der vollständigen CSV-Dateien pro
    Experiment und Klasse.
    """

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Dateien insgesamt: {len(records)}")

    experiment_counts = Counter(
        record.experiment
        for record in records
    )

    print("\nDateien pro Experiment:")

    for experiment, count in sorted(
        experiment_counts.items()
    ):
        print(f"  {experiment}: {count}")

    label_counts = Counter(
        record.label
        for record in records
    )

    print("\nDateien pro Klasse:")

    for class_name in CLASS_NAMES:
        print(
            f"  {class_name:10s}: "
            f"{label_counts.get(class_name, 0)}"
        )


# ============================================================
# GEMEINSAMEN SCALER ANPASSEN
# ============================================================

def fit_standard_scaler(
    training_records: list[FileRecord],
) -> StandardScaler:
    """
    Passt einen gemeinsamen StandardScaler ausschließlich
    an den kombinierten Trainingsdaten an.

    Jede der 256 Feature-Spalten erhält:

        Mittelwert ungefähr 0
        Standardabweichung ungefähr 1

    Validation und Test werden nicht für das Fitten verwendet.
    """

    scaler = StandardScaler()

    successfully_processed = 0
    failed_files = 0

    print("\n" + "=" * 70)
    print("Gemeinsamen StandardScaler anpassen")
    print("=" * 70)

    for file_number, record in enumerate(
        training_records,
        start=1,
    ):
        try:
            values = load_features(record.path)

        except Exception as error:
            failed_files += 1

            print(
                f"\nScaler: Datei übersprungen:\n"
                f"  {record.path}\n"
                f"  Grund: {error}"
            )
            continue

        # Dateiweise Anpassung, damit nicht alle Daten
        # gleichzeitig im Arbeitsspeicher liegen müssen.
        scaler.partial_fit(values)

        successfully_processed += 1

        if file_number % 100 == 0:
            print(
                f"Scaler-Fit: "
                f"{file_number}/{len(training_records)} Dateien"
            )

    if successfully_processed == 0:
        raise RuntimeError(
            "Der StandardScaler konnte nicht angepasst werden."
        )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    print(
        f"\nScaler mit {successfully_processed} Dateien angepasst."
    )
    print(f"Fehlgeschlagene Dateien: {failed_files}")
    print(f"Scaler gespeichert: {SCALER_PATH}")

    return scaler


# ============================================================
# STARTPOSITIONEN DER ZEITFENSTER
# ============================================================

def select_window_starts(
    number_of_packets: int,
) -> list[int]:
    """
    Bestimmt die Startpositionen der Zeitfenster.

    Beispiel bei ungefähr 220 Paketen:

        Start 0
        Start 64

    Beispiel bei 500 Paketen:

        Es wären normalerweise mehrere Fenster möglich.
        Wegen MAX_WINDOWS_PER_FILE werden aber nur maximal
        zwei möglichst weit auseinanderliegende Fenster gewählt.
    """

    if number_of_packets < WINDOW_SIZE:
        return []

    candidate_starts = list(
        range(
            0,
            number_of_packets - WINDOW_SIZE + 1,
            STEP_SIZE,
        )
    )

    if len(candidate_starts) <= MAX_WINDOWS_PER_FILE:
        return candidate_starts

    selected_indices = np.linspace(
        0,
        len(candidate_starts) - 1,
        num=MAX_WINDOWS_PER_FILE,
        dtype=np.int64,
    )

    selected_starts = [
        candidate_starts[index]
        for index in selected_indices
    ]

    # Doppelte Startpositionen entfernen
    return list(
        dict.fromkeys(selected_starts)
    )


# ============================================================
# ZEITFENSTER ERZEUGEN
# ============================================================

def create_windows(
    records: list[FileRecord],
    scaler: StandardScaler,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lädt, standardisiert und segmentiert alle Dateien eines Splits.

    Rückgabewerte:

        X:
            CNN-Eingaben mit Form
            (Anzahl Fenster, WINDOW_SIZE, 256)

        y:
            Numerische Klassen-IDs mit Form
            (Anzahl Fenster,)

        experiments:
            Herkunft jedes Fensters:
            2 für Experiment 2
            3 für Experiment 3
    """

    windows: list[np.ndarray] = []
    labels: list[int] = []
    experiment_ids: list[int] = []

    skipped_short = 0
    skipped_errors = 0

    print("\n" + "=" * 70)
    print(f"Fenster erzeugen: {split_name}")
    print("=" * 70)

    for file_number, record in enumerate(
        records,
        start=1,
    ):
        try:
            values = load_features(record.path)

        except Exception as error:
            skipped_errors += 1

            print(
                f"\nDatei übersprungen:\n"
                f"  {record.path}\n"
                f"  Grund: {error}"
            )
            continue

        start_positions = select_window_starts(
            number_of_packets=len(values)
        )

        if not start_positions:
            skipped_short += 1
            continue

        # Derselbe vorher angepasste Scaler wird für
        # Training, Validation und Test verwendet.
        scaled_values = scaler.transform(
            values
        ).astype(np.float32)

        label_id = LABEL_TO_ID[record.label]

        if record.experiment == "EXP2":
            experiment_id = 2
        else:
            experiment_id = 3

        for start in start_positions:
            end = start + WINDOW_SIZE

            window = scaled_values[
                start:end
            ]

            expected_shape = (
                WINDOW_SIZE,
                NUM_FEATURES,
            )

            if window.shape != expected_shape:
                skipped_errors += 1
                continue

            windows.append(window)
            labels.append(label_id)
            experiment_ids.append(experiment_id)

        if file_number % 100 == 0:
            print(
                f"Verarbeitet: "
                f"{file_number}/{len(records)} Dateien"
            )

    if not windows:
        raise RuntimeError(
            f"Für {split_name} wurden keine Fenster erzeugt."
        )

    X = np.stack(
        windows
    ).astype(np.float32)

    y = np.asarray(
        labels,
        dtype=np.int32,
    )

    experiments = np.asarray(
        experiment_ids,
        dtype=np.int8,
    )

    print(f"\nErzeugte Fenster: {len(X)}")
    print(f"X-Form: {X.shape}")
    print(f"y-Form: {y.shape}")
    print(f"Experiment-Form: {experiments.shape}")
    print(f"Zu kurze Dateien: {skipped_short}")
    print(f"Fehlerhafte Dateien/Fenster: {skipped_errors}")

    print("\nFenster pro Klasse:")

    class_distribution = Counter(
        y.tolist()
    )

    for class_name in CLASS_NAMES:
        class_id = LABEL_TO_ID[class_name]

        print(
            f"  {class_name:10s}: "
            f"{class_distribution.get(class_id, 0)}"
        )

    print("\nFenster pro Experiment:")

    experiment_distribution = Counter(
        experiments.tolist()
    )

    print(
        f"  Experiment 2: "
        f"{experiment_distribution.get(2, 0)}"
    )

    print(
        f"  Experiment 3: "
        f"{experiment_distribution.get(3, 0)}"
    )

    return X, y, experiments


# ============================================================
# DATENSATZ ALS NPZ SPEICHERN
# ============================================================

def create_and_save_split(
    records: list[FileRecord],
    scaler: StandardScaler,
    split_name: str,
    output_path: Path,
) -> None:
    """
    Erzeugt einen vollständigen Datensplit und speichert ihn
    als komprimierte NPZ-Datei.
    """

    X, y, experiments = create_windows(
        records=records,
        scaler=scaler,
        split_name=split_name,
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        experiments=experiments,
    )

    print(f"\nDatensatz gespeichert: {output_path}")

    # Speicher freigeben, bevor der nächste Split erzeugt wird.
    del X
    del y
    del experiments

    gc.collect()


# ============================================================
# GESPEICHERTE NPZ-DATEI KONTROLLIEREN
# ============================================================

def inspect_saved_dataset(
    name: str,
    path: Path,
) -> None:
    """
    Öffnet eine gespeicherte NPZ-Datei und zeigt deren Formen.
    """

    with np.load(path) as dataset:
        X = dataset["X"]
        y = dataset["y"]
        experiments = dataset["experiments"]

        print("\n" + "-" * 70)
        print(f"Kontrolle: {name}")
        print("-" * 70)

        print(f"X:           {X.shape} / {X.dtype}")
        print(f"y:           {y.shape} / {y.dtype}")
        print(
            f"experiments: {experiments.shape} / "
            f"{experiments.dtype}"
        )

        print(
            f"Endliche X-Werte: "
            f"{np.isfinite(X).all()}"
        )


# ============================================================
# PROGRAMMSTART
# ============================================================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # 1. Trainingsdateien beider Experimente finden
    # --------------------------------------------------------

    exp2_training_records = discover_records(
        root=EXP2_TRAIN_SOURCE,
        experiment="EXP2",
    )

    exp3_training_records = discover_records(
        root=EXP3_TRAIN_SOURCE,
        experiment="EXP3",
    )

    all_training_source_records = (
        exp2_training_records
        + exp3_training_records
    )

    # --------------------------------------------------------
    # 2. Internes Training und Validation erzeugen
    # --------------------------------------------------------

    train_records, validation_records = (
        split_training_records(
            all_training_source_records
        )
    )

    # --------------------------------------------------------
    # 3. Vorhandene Validation-Ordner als Test verwenden
    # --------------------------------------------------------

    exp2_test_records = discover_records(
        root=EXP2_TEST_SOURCE,
        experiment="EXP2",
    )

    exp3_test_records = discover_records(
        root=EXP3_TEST_SOURCE,
        experiment="EXP3",
    )

    test_records = (
        exp2_test_records
        + exp3_test_records
    )

    # --------------------------------------------------------
    # 4. Verteilungen anzeigen
    # --------------------------------------------------------

    print_record_distribution(
        name="Neuer Trainingsdatensatz",
        records=train_records,
    )

    print_record_distribution(
        name="Interner Validierungsdatensatz",
        records=validation_records,
    )

    print_record_distribution(
        name="Unabhängiger Testdatensatz",
        records=test_records,
    )

    # --------------------------------------------------------
    # 5. Dateilisten zur Reproduzierbarkeit speichern
    # --------------------------------------------------------

    save_manifest(
        records=train_records,
        filename="train_manifest.csv",
    )

    save_manifest(
        records=validation_records,
        filename="validation_manifest.csv",
    )

    save_manifest(
        records=test_records,
        filename="test_manifest.csv",
    )

    # --------------------------------------------------------
    # 6. Scaler nur auf Trainingsdaten anpassen
    # --------------------------------------------------------

    scaler = fit_standard_scaler(
        training_records=train_records
    )

    # --------------------------------------------------------
    # 7. CNN-Datensätze erzeugen und speichern
    # --------------------------------------------------------

    create_and_save_split(
        records=train_records,
        scaler=scaler,
        split_name="Training",
        output_path=TRAIN_DATA_PATH,
    )

    create_and_save_split(
        records=validation_records,
        scaler=scaler,
        split_name="Validation",
        output_path=VALIDATION_DATA_PATH,
    )

    create_and_save_split(
        records=test_records,
        scaler=scaler,
        split_name="Test",
        output_path=TEST_DATA_PATH,
    )

    # --------------------------------------------------------
    # 8. Gespeicherte Dateien kontrollieren
    # --------------------------------------------------------

    inspect_saved_dataset(
        name="Training",
        path=TRAIN_DATA_PATH,
    )

    inspect_saved_dataset(
        name="Validation",
        path=VALIDATION_DATA_PATH,
    )

    inspect_saved_dataset(
        name="Test",
        path=TEST_DATA_PATH,
    )

    print("\n" + "=" * 70)
    print("DATENAUFBEREITUNG ABGESCHLOSSEN")
    print("=" * 70)

    print(f"Training:   {TRAIN_DATA_PATH}")
    print(f"Validation: {VALIDATION_DATA_PATH}")
    print(f"Test:       {TEST_DATA_PATH}")
    print(f"Scaler:     {SCALER_PATH}")


if __name__ == "__main__":
    main()