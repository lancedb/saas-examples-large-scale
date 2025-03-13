import pandas as pd

# Read the artwork.csv file
df = pd.read_csv('data/artworks.csv')

# Count non-null and null entries in primaryImage column
non_null_count = df['image_url'].count()
null_count = df['image_url'].isnull().sum()
total_count = len(df)

print(f"Number of non-null entries in primaryImage: {non_null_count}")
print(f"Number of null/missing entries in primaryImage: {null_count}")
print(f"Total number of rows: {total_count}")
print(f"Percentage of non-null entries: {(non_null_count/total_count)*100:.2f}%")
print(f"Percentage of null entries: {(null_count/total_count)*100:.2f}%")
