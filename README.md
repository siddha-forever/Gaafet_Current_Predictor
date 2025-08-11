# GAAFET *I<sub>d</sub>* and Charge Predictor

A **machine learning-based predictive tool** for estimating the **drain current (*I<sub>d</sub>*)** and **charge** in a **Gate-All-Around Field-Effect Transistor (GAAFET)** under various radiation strike conditions.  
This application provides an interactive interface to visualize *I<sub>d</sub>* vs. time and the corresponding integrated charge vs. time.

---

## 🚀 Features

- **Interactive Parameter Inputs**  
  - **Phi (°)**: Angle of radiation strike with respect to device geometry.  
  - **Theta (°)**: Azimuthal angle of the strike.  
  - **LET (°)**: Linear Energy Transfer value in MeV·cm²/mg.

- **Real-time Predictions**
  - **Predicted *I<sub>d</sub>* vs. Time** plot.
  - **Charge vs. Time** (integrated *I<sub>d</sub>*) plot.

- **Data Tables**
  - Option to view the **current prediction table**.
  - Option to view the **charge table**.

---

## 📊 Example Output

| Input Parameter | Value |
|-----------------|-------|
| Phi (°)         | 70.00 |
| Theta (°)       | 180.00 |
| LET (°)         | 120.00 |

**Predicted *I<sub>d</sub>* vs. Time**
![Predicted Id vs Time](path/to/id_plot.png)

**Charge vs. Time**
![Charge vs Time](path/to/charge_plot.png)

---

## 🛠️ Technology Stack

- **Frontend & Interface**: [Streamlit](https://streamlit.io/)  
- **Backend Model**: Pre-trained machine learning model for *I<sub>d</sub>* prediction.  
- **Languages**: Python  
- **Visualization**: Matplotlib / Pandas

---
## 📬 Contact
For questions, feedback, or collaboration opportunities, feel free to reach out:

- GitHub : https://github.com/siddha-forever
- Email : mohapatra.siddhabrata@gmail.com
- LinkedIn : [https://www.linkedin.com/in/siddhabrata-mohapatra](https://www.linkedin.com/in/siddhabrata-mohapatra)

---

## 🙏 Acknowledgments
- Dataset source: Custom build
- Mentor - Faculty: Dr. Biswajit Jena
