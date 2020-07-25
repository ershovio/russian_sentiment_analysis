import os


def change_separator_in_csv(
        old_file_path: str,
        new_file_path: str,
        old_sep: str = ",",
        new_sep: str = "&"
):
    """
    Replaces the last occurrence of `old_sep` with `new_sep`
    in all rows of `old_file_path` csv file and saves new file in `new_file_path`

    This prepossessing is needed because the original files contain
    different number of commas in rows.
    """
    if not os.path.isfile(old_file_path):
        raise ValueError(f"File {old_file_path} does not exist")
    with open(old_file_path, "r") as old_csv:
        with open(new_file_path, "w") as new_csv:
            for old_row in old_csv:
                sep_ind = old_row.rfind(old_sep)
                new_row = old_row[:sep_ind] + new_sep + old_row[sep_ind + 1:]
                new_csv.write(f"{new_row}\n")

