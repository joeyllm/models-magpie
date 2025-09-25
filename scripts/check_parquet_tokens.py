# scripts/check_parquet_tokens.py
import sys, pyarrow.dataset as ds, pyarrow as pa

def is_int_list(field: pa.Field) -> bool:
    return pa.types.is_list(field.type) and pa.types.is_integer(field.type.value_type)

def main(root="outputs/data"):
    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    schema = dataset.schema
    cols = {f.name: f for f in schema}

    has_input_ids = "input_ids" in cols and is_int_list(cols["input_ids"])
    has_attn = "attention_mask" in cols and is_int_list(cols["attention_mask"])
    has_labels = ("labels" in cols and is_int_list(cols["labels"])) or ("target_ids" in cols and is_int_list(cols["target_ids"]))

    print("Schema:", schema)
    print(f"input_ids: {has_input_ids}, attention_mask: {has_attn}, labels/target_ids: {has_labels}")

    if not has_input_ids:
        print("❌ Not tokenized: missing `input_ids`.")
        sys.exit(1)

    # sample a few rows to sanity check lengths and value types
    tbl = dataset.to_table(columns=[c for c in ["input_ids","attention_mask","labels","target_ids"] if c in cols], limit=8)
    for i in range(min(3, tbl.num_rows)):
        row = {name: tbl.column(name)[i].as_py() if name in tbl.column_names else None for name in ["input_ids","attention_mask","labels","target_ids"]}
        li = row["input_ids"]
        if not isinstance(li, list) or not all(isinstance(x, int) for x in li):
            print("❌ `input_ids` is not a list[int].")
            sys.exit(1)
        am = row.get("attention_mask")
        if am is not None and len(am) != len(li):
            print("⚠️ attention_mask length != input_ids length (row", i, ")")
    print("✅ Looks tokenized.")
    sys.exit(0)

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs/data"
    main(root)

