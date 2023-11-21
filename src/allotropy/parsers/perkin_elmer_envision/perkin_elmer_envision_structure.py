# plate (repeated for N plates measured)
#     Plate information
#     Background information (optional)
#     Calculated results (optional)
#     Measured results (repeated for N measurements performed per plate)
# Basic assay information
# Protocol information
# Plate type
# Platemap (repeated for N plates measured)
# Calculations
# Auto export parameters
# Operations
# Labels
#     Filters (dependent on detection modality)
#     Mirrors (dependent on detection modality)
# Instrument
from __future__ import annotations

from dataclasses import dataclass
from re import search
from typing import Optional, Union

import numpy as np
import pandas as pd

from allotropy.allotrope.allotrope import AllotropyError
from allotropy.allotrope.models.plate_reader_benchling_2023_09_plate_reader import (
    ScanPositionSettingPlateReader,
)
from allotropy.allotrope.models.shared.components.plate_reader import SampleRoleType
from allotropy.parsers.lines_reader import CsvReader
from allotropy.parsers.utils.values import assert_not_none, try_float


def df_to_series(df: pd.DataFrame) -> pd.Series:
    df.columns = df.iloc[0]  # type: ignore[assignment]
    return pd.Series(df.iloc[-1], index=df.columns)


def str_from_series(
    data: pd.Series, key: str, default: Optional[str] = None
) -> Optional[str]:
    value = data.get(key, default)
    return None if value is None else str(value)


def assert_float(value: Optional[str], name: Optional[str] = None) -> float:
    return assert_not_none(
        try_float(value),
        msg=f"Expected float value{f' for {name}' if name else ''}",
    )


def float_from_series(data: pd.Series, key: str) -> Optional[float]:
    try:
        value = data.get(key)
        return try_float(str(value))
    except Exception as e:
        msg = f"Unable to convert {key} to float value"
        raise AllotropyError(msg) from e


def assert_float_from_series(
    data: pd.Series, key: str, msg: Optional[str] = None
) -> float:
    return assert_not_none(float_from_series(data, key), key, msg)


def num_to_chars(n: int) -> str:
    d, m = divmod(n, 26)  # 26 is the number of ASCII letters
    return "" if n < 0 else num_to_chars(d - 1) + chr(m + 65)  # chr(65) = 'A'


@dataclass
class PlateInfo:
    number: str
    barcode: str
    measurement_time: Optional[str]
    measured_height: Optional[float]
    chamber_temperature_at_start: Optional[float]

    @staticmethod
    def get_series(reader: CsvReader) -> pd.Series:
        assert_not_none(
            reader.pop_if_match("^Plate information"),
            msg="Unable to find expected plate information",
        )

        data = assert_not_none(
            reader.pop_csv_block_as_df(),
            "Plate information CSV block",
        )
        series = df_to_series(data).replace(np.nan, None)
        series.index = pd.Series(series.index).replace(np.nan, "empty label")  # type: ignore[assignment]
        return series

    @staticmethod
    def get_plate_number(series: pd.Series) -> str:
        return assert_not_none(
            str_from_series(series, "Plate"),
            "Plate information: Plate",
        )

    @staticmethod
    def get_barcode(series: pd.Series, plate_number: str) -> str:
        raw_barcode = str_from_series(series, "Barcode") or '=""'
        barcode = raw_barcode.removeprefix('="').removesuffix('"')
        return barcode or f"Plate {plate_number}"


@dataclass
class CalculatedPlateInfo(PlateInfo):
    formula: str

    @staticmethod
    def create(series: pd.Series) -> Optional[CalculatedPlateInfo]:
        plate_number = PlateInfo.get_plate_number(series)
        return CalculatedPlateInfo(
            plate_number,
            barcode=PlateInfo.get_barcode(series, plate_number),
            measurement_time=str_from_series(series, "Measurement date"),
            measured_height=float_from_series(series, "Measured height"),
            chamber_temperature_at_start=float_from_series(
                series,
                "Chamber temperature at start",
            ),
            formula=assert_not_none(
                str_from_series(series, "Formula"),
                msg="Unable to get expected formula for calculated results section",
            ),
        )


@dataclass
class ResultPlateInfo(PlateInfo):
    label: str
    emission_filter_id: str

    @staticmethod
    def create(series: pd.Series) -> Optional[ResultPlateInfo]:
        plate_number = PlateInfo.get_plate_number(series)
        barcode = PlateInfo.get_barcode(series, plate_number)

        label = str_from_series(series, "Label")
        if label is None:
            return None

        measinfo = str_from_series(series, "Measinfo")
        if measinfo is None:
            return None

        emission_id_search_result = assert_not_none(
            search("De=(...)", measinfo),
            msg=f"Unable to get emition filter id from plate {barcode}",
        )

        return ResultPlateInfo(
            plate_number,
            barcode,
            measurement_time=str_from_series(series, "Measurement date"),
            measured_height=float_from_series(series, "Measured height"),
            chamber_temperature_at_start=float_from_series(
                series,
                "Chamber temperature at start",
            ),
            label=label,
            emission_filter_id=emission_id_search_result.group(1),
        )


@dataclass
class CalculatedResult:
    col: str
    row: str
    value: float


@dataclass
class CalculatedResultList:
    calculated_results: list[CalculatedResult]

    @staticmethod
    def create(reader: CsvReader) -> CalculatedResultList:
        # Calculated results may or may not have a title
        reader.pop_if_match("^Calculated results")

        data = assert_not_none(
            reader.pop_csv_block_as_df(),
            "results data",
        )
        series = (
            data.drop(0, axis=0).drop(0, axis=1) if data.iloc[1, 0] == "A" else data
        )
        rows, cols = series.shape
        series.index = [num_to_chars(i) for i in range(rows)]  # type: ignore[assignment]
        series.columns = [str(i).zfill(2) for i in range(1, cols + 1)]  # type: ignore[assignment]

        return CalculatedResultList(
            calculated_results=[
                CalculatedResult(col, row, series.loc[col, row])
                for col, row in series.stack().index
            ]
        )


@dataclass
class Result:
    col: str
    row: str
    value: int


@dataclass
class ResultList:
    results: list[Result]

    @staticmethod
    def create(reader: CsvReader) -> ResultList:
        # Results may or may not have a title
        reader.pop_if_match("^Results")

        data = assert_not_none(
            reader.pop_csv_block_as_df(),
            "results data",
        )
        series = (
            data.drop(0, axis=0).drop(0, axis=1) if data.iloc[1, 0] == "A" else data
        )
        rows, cols = series.shape
        series.index = [num_to_chars(i) for i in range(rows)]  # type: ignore[assignment]
        series.columns = [str(i).zfill(2) for i in range(1, cols + 1)]  # type: ignore[assignment]

        return ResultList(
            results=[
                Result(col, row, int(series.loc[col, row]))
                for col, row in series.stack().index
            ]
        )


@dataclass
class Plate:
    plate_info: Union[CalculatedPlateInfo, ResultPlateInfo]
    calculated_results: CalculatedResultList
    results: ResultList

    @staticmethod
    def create(reader: CsvReader) -> list[Plate]:
        plates: list[Plate] = []
        while reader.match("^Plate information"):
            series = PlateInfo.get_series(reader)
            if result_plate_info := ResultPlateInfo.create(series):
                reader.drop_sections("^Background information")
                plates.append(
                    Plate(
                        result_plate_info,
                        calculated_results=CalculatedResultList([]),
                        results=ResultList.create(reader),
                    )
                )
            elif calculated_plate_info := CalculatedPlateInfo.create(series):
                reader.drop_sections("^Background information")
                plates.append(
                    Plate(
                        calculated_plate_info,
                        calculated_results=CalculatedResultList.create(reader),
                        results=ResultList([]),
                    )
                )
            else:
                msg = "Unable to interpret plate information"
                raise AllotropyError(msg)
        return plates


@dataclass
class BasicAssayInfo:
    protocol_id: Optional[str]
    assay_id: Optional[str]

    @staticmethod
    def create(reader: CsvReader) -> BasicAssayInfo:
        reader.drop_until_inclusive("^Basic assay information")
        data = assert_not_none(
            reader.pop_csv_block_as_df(),
            "Basic assay information",
        )
        data = data.T
        data.iloc[0].replace(":.*", "", regex=True, inplace=True)
        series = df_to_series(data)
        return BasicAssayInfo(
            str_from_series(series, "Protocol ID"),
            str_from_series(series, "Assay ID"),
        )


@dataclass
class PlateType:
    number_of_wells: float

    @staticmethod
    def create(reader: CsvReader) -> PlateType:
        reader.drop_until_inclusive("^Plate type")
        data = assert_not_none(
            reader.pop_csv_block_as_df(),
            "Plate type",
        )
        series = df_to_series(data.T)
        return PlateType(
            assert_float_from_series(series, "Number of the wells in the plate")
        )


def get_sample_role_type(encoding: str) -> SampleRoleType:
    # BL        blank               blank_role
    # CTL       control             control_sample_role
    # LB        lance_blank         blank_role
    # LC        lance_crosstalk     control_sample_role
    # LH        lance_high          control_sample_role
    # S         pl_sample           sample_role
    # STD       standard            standard_sample_role
    # -         unknown             unknown_sample_role
    # UNK       unknown             unknown_sample_role
    # ZH        z_high              control_sample_role
    # ZL        z_low               control_sample_role
    sample_role_type_map = {
        "BL": SampleRoleType.blank_role,
        "CTL": SampleRoleType.control_sample_role,
        "LB": SampleRoleType.blank_role,
        "LC": SampleRoleType.control_sample_role,
        "LH": SampleRoleType.control_sample_role,
        "STD": SampleRoleType.standard_sample_role,
        "S": SampleRoleType.sample_role,
        "-": SampleRoleType.unknown_sample_role,
        "UNK": SampleRoleType.unknown_sample_role,
        "ZH": SampleRoleType.control_sample_role,
        "ZL": SampleRoleType.control_sample_role,
    }
    for pattern, value in sample_role_type_map.items():
        if encoding.startswith(pattern):
            return value

    msg = f"Unable to determine sample role type of plate map encoding {encoding}"
    raise ValueError(msg)


@dataclass
class PlateMap:
    plate_n: str
    group_n: str
    sample_role_type_mapping: dict[str, dict[str, SampleRoleType]]

    @staticmethod
    def create(reader: CsvReader) -> Optional[PlateMap]:
        if not reader.current_line_exists() or reader.match("^Calculations"):
            return None

        plate_n = assert_not_none(reader.pop(), "Platemap number").split(",")[-1]
        group_n = assert_not_none(reader.pop(), "Platemap group").split(",")[-1]

        data = assert_not_none(
            reader.pop_csv_block_as_df(),
            "Platemap data",
        ).replace(" ", "", regex=True)

        reader.pop_data()  # drop type specification
        reader.drop_empty()

        series = (
            data.drop(0, axis=0).drop(0, axis=1) if data.iloc[1, 0] == "A" else data
        )
        rows, cols = series.shape
        series.index = [num_to_chars(i) for i in range(rows)]  # type: ignore[assignment]
        series.columns = [str(i).zfill(2) for i in range(1, cols + 1)]  # type: ignore[assignment]

        sample_role_type_mapping: dict[str, dict[str, SampleRoleType]] = {}
        for row, row_data in series.replace([np.nan, "''"], None).to_dict().items():
            col_mapping: dict[str, SampleRoleType] = {}
            for col, value in row_data.items():
                if value:
                    if role_type := get_sample_role_type(str(value)):
                        col_mapping[str(col)] = role_type
            if col_mapping:
                sample_role_type_mapping[str(row)] = col_mapping

        return PlateMap(plate_n, group_n, sample_role_type_mapping)

    def get_sample_role_type(self, col: str, row: str) -> SampleRoleType:
        try:
            return self.sample_role_type_mapping[row][col]
        except KeyError as e:
            msg = (
                f"Invalid plate map location for plate map {self.plate_n}: {col} {row}"
            )
            raise AllotropyError(msg) from e


def create_plate_maps(reader: CsvReader) -> dict[str, PlateMap]:
    assert_not_none(
        reader.drop_until_inclusive("^Platemap"),
        msg="Unable to get plate map information",
    )

    maps: dict[str, PlateMap] = {}
    while _map := PlateMap.create(reader):
        maps[_map.plate_n] = _map
    return maps


@dataclass
class Filter:
    name: str
    wavelength: float
    bandwidth: Optional[float] = None

    @staticmethod
    def create(reader: CsvReader) -> Optional[Filter]:
        if not reader.current_line_exists() or reader.match(
            "(^Mirror modules)|(^Instrument:)|(^Aperture:)"
        ):
            return None

        data = assert_not_none(
            reader.pop_csv_block_as_df(),
            "Filter information",
        )
        series = df_to_series(data.T)

        name = str(series.index[0])

        description = str_from_series(series, "Description") or ""

        search_result = search("(Longpass)=\\d*nm", description)
        if search_result is not None:
            wavelength = float(
                search_result.group().removeprefix("Longpass=").removesuffix("nm")
            )
            return Filter(name, wavelength)

        search_result = assert_not_none(
            search("(CWL)=\\d*nm", description),
            msg=f"Unable to find wavelength for filter {name}",
        )
        wavelength = float(
            search_result.group().removeprefix("CWL=").removesuffix("nm")
        )
        search_result = assert_not_none(
            search("BW=\\d*nm", description),
            msg=f"Unable to find bandwidth for filter {name}",
        )
        bandwidth = float(search_result.group().removeprefix("BW=").removesuffix("nm"))

        return Filter(name, wavelength, bandwidth=bandwidth)


def create_filters(reader: CsvReader) -> dict[str, Filter]:
    reader.drop_until("(^Filters:)|^Instrument:")

    if reader.match("^Instrument"):
        return {}

    reader.pop()  # remove title

    filters = {}
    while _filter := Filter.create(reader):
        filters[_filter.name] = _filter
    return filters


@dataclass
class Labels:
    label: str
    excitation_filter: Optional[Filter]
    emission_filters: dict[str, Optional[Filter]]
    scan_position_setting: Optional[ScanPositionSettingPlateReader] = None
    number_of_flashes: Optional[float] = None
    detector_gain_setting: Optional[str] = None

    @staticmethod
    def create(reader: CsvReader) -> Labels:
        reader.drop_until_inclusive("^Labels")
        data = assert_not_none(
            reader.pop_csv_block_as_df(),
            "Labels",
        )
        series = df_to_series(data.T).replace(np.nan, None)

        filters = create_filters(reader)
        filter_position_map = {
            "Bottom": ScanPositionSettingPlateReader.bottom_scan_position__plate_reader_,
            "Top": ScanPositionSettingPlateReader.top_scan_position__plate_reader_,
        }

        return Labels(
            series.index[0],
            excitation_filter=filters.get(str_from_series(series, "Exc. filter") or ""),
            emission_filters={
                "1st": filters.get(str_from_series(series, "Ems. filter") or ""),
                "2nd": filters.get(str_from_series(series, "2nd ems. filter") or ""),
            },
            scan_position_setting=filter_position_map.get(
                str_from_series(series, "Using of emission filter") or ""
            ),
            number_of_flashes=float_from_series(series, "Number of flashes"),
            detector_gain_setting=str_from_series(series, "Reference AD gain"),
        )

    def get_emission_filter(self, id_val: str) -> Optional[Filter]:
        return self.emission_filters.get(id_val)


@dataclass
class Instrument:
    serial_number: str
    nickname: str

    @staticmethod
    def create(reader: CsvReader) -> Instrument:
        assert_not_none(
            reader.drop_until_inclusive("^Instrument"),
            msg="Unable to find instrument information",
        )

        serial_number = assert_not_none(reader.pop(), "serial number").split(",")[-1]
        nickname = assert_not_none(reader.pop(), "nickname").split(",")[-1]

        return Instrument(serial_number, nickname)


@dataclass
class Data:
    plates: list[Plate]
    basic_assay_info: BasicAssayInfo
    number_of_wells: float
    plate_maps: dict[str, PlateMap]
    labels: Labels
    instrument: Instrument

    @staticmethod
    def create(reader: CsvReader) -> Data:
        return Data(
            plates=Plate.create(reader),
            basic_assay_info=BasicAssayInfo.create(reader),
            number_of_wells=PlateType.create(reader).number_of_wells,
            plate_maps=create_plate_maps(reader),
            labels=Labels.create(reader),
            instrument=Instrument.create(reader),
        )
