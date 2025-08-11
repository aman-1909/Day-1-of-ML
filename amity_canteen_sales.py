import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1.Load data
df = pd.read_csv("/amity_canteen_sales.csv")

# 2.Convert days name into numerical code
day_mapping = {"Mon": 0, "Tue": 1, "Wed": 2,
               "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
df["Day_Num"] = df["Day"].map(day_mapping)

#3.Model training
X = df[["Day_Num", "Temperature", "Special_Event"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

# 4.Get info and predict
print("📅 College Canteen Sales Predictor")
day_input = input("Enter day (Mon/Tue/Wed/Thu/Fri/Sat/Sun): ")
temp_input = float(input("Enter temperature (°C): "))
event_input = int(input("Special event? (0=No, 1=Yes): "))

# 6.Convert day to number
if day_input not in day_mapping:
    print("❌ Invalid day entered.")
else:
    day_num = day_mapping[day_input]
    new_data = [[day_num, temp_input, event_input]]
    predicted_sales = model.predict(new_data)
    print(f"💰 Predicted Sales: ₹{predicted_sales[0]:.2f}")

# 7. Visualization on Scatter graph
plt.scatter(df["Temperature"], df["Sales"], c=df["Special_Event"], cmap='coolwarm')
plt.xlabel("Temperature (°C)")
plt.ylabel("Sales (₹)")
plt.title("Canteen Sales vs Temperature")
plt.colorbar(label="Special Event (0=No, 1=Yes)")
plt.show()