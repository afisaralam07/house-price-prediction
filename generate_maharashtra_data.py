import pandas as pd
import numpy as np
import random

np.random.seed(42)

cities_price_per_sqft = {
    "Mumbai": 28000,
    "Thane": 18000,
    "Pune": 12000,
    "Nashik": 6000,
    "Nagpur": 5500,
    "Aurangabad": 5000,
    "Kolhapur": 4800,
    "Solapur": 4200
}

data = []

for _ in range(5000):

    city = random.choice(list(cities_price_per_sqft.keys()))
    base_rate = cities_price_per_sqft[city]

    area = np.random.randint(400, 3000)
    bedrooms = np.random.randint(1, 5)
    bathrooms = np.random.randint(1, 4)
    age = np.random.randint(0, 35)
    parking = np.random.randint(0, 3)
    furnishing = random.choice(["Unfurnished", "Semi-Furnished", "Fully Furnished"])
    air_conditioning = random.choice(["Yes", "No"])
    main_road = random.choice(["Yes", "No"])

    # Base price from area
    price = area * base_rate

    # Bedrooms & bathrooms impact
    price += bedrooms * 300000
    price += bathrooms * 200000

    # Age depreciation
    price -= age * 60000

    # Furnishing premium
    if furnishing == "Fully Furnished":
        price += 500000
    elif furnishing == "Semi-Furnished":
        price += 250000

    # Parking value
    price += parking * 200000

    # AC premium
    if air_conditioning == "Yes":
        price += 300000

    # Main road premium
    if main_road == "Yes":
        price += 400000

    # Market fluctuation noise
    noise = np.random.normal(0, 0.05 * price)
    price += noise

    # Ensure minimum price
    price = max(price, 800000)

    data.append([
        round(price),
        city,
        area,
        age,
        bedrooms,
        bathrooms,
        furnishing,
        parking,
        air_conditioning,
        main_road
    ])

columns = [
    "price",
    "location",
    "area",
    "property_age",
    "bedrooms",
    "bathrooms",
    "furnishing",
    "parking",
    "air_conditioning",
    "main_road"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("maharashtra_housing_data.csv", index=False)

print("Maharashtra Housing Dataset Created Successfully!")