import numpy as np
import matplotlib.pyplot as plt

# Fixed sensitivity
sensitivity = 0.99

# Specificity values
specificities = [0.99, 0.999, 0.9999, 0.99999]

# Prevalence range (0.001% to 50%)
prevalence = np.linspace(0.00001, 0.5, 500)

plt.figure(figsize=(8,6))

for spec in specificities:
    
    # False positive rate
    fpr = 1 - spec
    
    # Positive Predictive Value (probability infected given positive test)
    ppv = (sensitivity * prevalence) / (
        sensitivity * prevalence + fpr * (1 - prevalence)
    )
    
    plt.plot(prevalence * 100, ppv * 100, label=f"Specificity {spec*100}%")

plt.xlabel("Infection Prevalence (%)")
plt.ylabel("Probability Fred is Infected (%)")
plt.title("Probability of True Infection After Positive Test")
plt.legend()
plt.grid(True)

plt.show()
