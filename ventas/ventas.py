from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wikisql")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wikisql")

xlsx_file = 'ventas2.xlsx'
df = pd.read_excel(xlsx_file)

data = {
    "cliente": df["cliente"].tolist(),
    "vendedor": df["vendedor"].tolist(),
    "articulo": df["articulo"].tolist(),
    "cantidad": df["cantidad"].tolist(),
    "precio": df["precio"].tolist(),
}

table = pd.DataFrame.from_dict(data)

# tapex accepts uncased input since it is pre-trained on the uncased corpus
query = "cliente, vendedor, articulo mas vendido por Marcos"
encoding = tokenizer(table=table, query=query, return_tensors="pt")

outputs = model.generate(**encoding)
print(query)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
