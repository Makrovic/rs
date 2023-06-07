import numpy as np
import pandas as pd
import regex as re
from currency_converter import CurrencyConverter

c = CurrencyConverter()

df = pd.read_csv("laptop_price.csv", encoding="latin-1")

df = df.drop("laptop_ID", axis=1)
df = df.rename(columns=str.lower)

# ram
df["ram"] = df["ram"].replace("[GB]", "", regex=True)
df["ram"] = pd.to_numeric(df["ram"])

# weight
df["weight"] = df["weight"].replace("[kg]", "", regex=True)
df["weight"] = pd.to_numeric(df["weight"])

# storage type & memory 1
df["memory"] = df["memory"].str.replace("1.0TB", "1TB", regex=True)
df["memory"] = df["memory"].str.replace("1TB", "1000GB")
df["memory"] = df["memory"].str.replace("2TB", "2000GB")
df["memory"] = df["memory"].str.replace("GB", "")

split_mem = df["memory"].str.split(" ", n=1, expand=True)
df["storage_type"] = split_mem[1]
df["storage_type"] = df["storage_type"].str.replace(r" ", "")
df["storage_1"] = split_mem[0]

# memory 2
memory_1 = []
memory_2 = []
for i in df["storage_type"]:
    if len(re.findall(r"\+", i)) == 1:  # DOUBLE DRIVE
        one = re.findall(r"([0-9]+)", i)
        memory_2.append(one[0])
    else:  # SINGLE DRIVE
        one = re.findall(r"(\w+)", i)
        memory_2.append("NaN")

df["storage_2"] = memory_2
df["storage_type"] = df["storage_type"].str.replace(r"([0-9]+)", "", regex=True)
df = df.drop("memory", axis=1)
df[["storage_type", "storage_2"]].value_counts()

# cpu speed
df["cpu_speed"] = df["cpu"].str.extract(r"(\d+(?:\.\d+)?GHz)")
df["cpu_speed"] = df["cpu_speed"].replace("[GHz]", "", regex=True)
df["cpu_speed"] = df["cpu_speed"].astype(float)

# cpu vendor
split_vendor = df["cpu"].str.split(" ", n=1, expand=True)
df["cpu_vendor"] = split_vendor[0]
df["cpu"] = split_vendor[1]

df["cpu"] = df["cpu"].str.replace(r"(\d+(?:\.\d+)?GHz)", "", regex=True)
df.rename(columns={"cpu": "cpu_model"}, inplace=True)

# screen resolution
temp_reso = df["screenresolution"].str.split(" ")
df["reso"] = temp_reso.str.get(-1)

df[["screen_width", "screen_height"]] = df["reso"].str.split("x", expand=True)

df["screen_type"] = df["screenresolution"].replace(r"(\d+x\d+)", "", regex=True)
df["screen_type"] = df["screen_type"].replace(
    r"(Full HD|Quad HD|Quad HD|\+|/|4K Ultra HD)", "", regex=True
)

df = df.drop("screenresolution", axis=1)
df = df.drop("reso", axis=1)

# touchscreen
df["touch_screen"] = df["screen_type"].str.extract(r"(Touchscreen)")
df["screen_type"] = df["screen_type"].replace(r"(Touchscreen)", "", regex=True)
df["touch_screen"] = df["touch_screen"].replace("Touchscreen", 1)
df["touch_screen"] = df["touch_screen"].replace(np.nan, 0)

# screen type
df["screen_type"] = df["screen_type"].replace(r"^\s*$", "Unspecified", regex=True)

# gpu
temp_df001 = df["gpu"].str.split()
df["gpu_vendor"] = temp_df001.str.get(0)

df_temp002 = list(df["gpu"].str.split())
df_temp002

df_temp003 = []
for i in df_temp002:
    df_temp003.append(" ".join(i[1:]))
df_temp003

df["gpu_model"] = df_temp003

df = df.drop("gpu", axis=1)
df[["gpu_vendor", "gpu_model"]].value_counts()

# price
df["price"] = df.apply(lambda x: c.convert(x.price_euros, "EUR", "IDR"), axis=1).astype(
    int
)
df = df.drop("price_euros", axis=1)


# clean
df.rename(columns={"typename": "type_name"}, inplace=True)
df.rename(columns={"opsys": "os"}, inplace=True)

df2 = df[
    [
        "company",
        "product",
        "type_name",
        "inches",
        "weight",
        "cpu_vendor",
        "cpu_model",
        "cpu_speed",
        "ram",
        "storage_type",
        "storage_1",
        "storage_2",
        "os",
        "screen_width",
        "screen_height",
        "screen_type",
        "touch_screen",
        "gpu_vendor",
        "gpu_model",
        "price",
    ]
]
df2.to_excel("output.xlsx")
print(df2)
