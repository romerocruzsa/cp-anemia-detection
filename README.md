# Efficient Neural Compression for Biomedical AI: Enabling Low-Power Non-Invasive Anemia Screening
*Alt. Project Titles: Addressing Memory Consumption through Data Quantization; Development of a Hemoglobin Estimation & Point-of-Care Diagnostic Tool for Anemia Detection*

## 📖 **Project Overview**  

Anemia is a prevalent and debilitating condition that affects approximately 1.62 billion people worldwide, with a disproportionate impact on individuals in low- and middle-income countries (LMICs). It is characterized by a deficiency of red blood cells or hemoglobin, leading to reduced oxygen transport and systemic hypoxia, which can contribute to the development of various chronic conditions. Moreover, the burden of anemia is significantly higher in LMICs due to malnutrition, infectious diseases such as malaria and hookworm infections, and limited access to healthcare service.

<p align="center">
  <img width="700" alt="Screenshot 2025-03-18 at 4 52 41 PM" 
  src="https://github.com/user-attachments/assets/d8a44c21-1b62-4c40-b901-6d8f677b1965">
</p>

Individuals in these regions often face economic barriers that hinder their ability to obtain timely diagnostic testing and treatment. In many rural or underserved areas, laboratory-based blood tests, considered the gold standard for anemia detection, are costly and logistically challenging due to inadequate medical infrastructure, a shortage of trained healthcare professionals, and inefficient distribution of medical supplies.

However, point-of-care (PoC) diagnostic tools have revolutionized medical diagnostics by enabling rapid, on-site testing without the need for centralized laboratories. These devices include pulse oximeters, thermography scanners, and optical-based hemoglobin measurement devices, which provide quick assessments without requiring blood samples. While these technologies are widely used in clinical and at-home settings, they present several inefficiencies that limit their applicability, particularly in low-resource environments. The lack of sustainability in the design, development, and disposal of medical devices presents a growing challenge in the healthcare industry. Additionally, the high cost of proprietary PoC devices limits widespread adoption, while their lack of interoperability with other medical technologies constrains integration into existing healthcare infrastructures.

### *To address these challenges, there is a need for development of a sustainable, AI-driven diagnostic solution that leverages repurposable hardware, minimizes electronic waste, and ensures longevity through software updates and adaptable designs co-designed with LMICs in mind.*

<p align="center">
  <img width="700" alt="Screenshot 2025-03-18 at 4 52 41 PM" 
  src="https://github.com/user-attachments/assets/e82a7a87-464b-46bf-a8a5-e98466f1440d">
</p>

By optimizing existing mobile devices and embedded systems for medical diagnostics, a more environmentally friendly and cost-effective alternative can be developed, fostering greater accessibility to healthcare services, particularly in underserved regions. We aim to survey different architectures to serve as a backbone fine-tune a hybrid-model end-to-end for estimating hemoglobin levels and classify sample severity. To optimize for embedded device deployment we propose applying different model compression techniques (Quantization, Pruning, Knowldege Distillation, etc.) for efficient compute, constrained-energy consumption, and remote-connection independence.

<p align="center">
  <img width="650" alt="Screenshot 2025-03-18 at 4 52 41 PM" 
  src="https://github.com/user-attachments/assets/ac66e01f-11d7-4bfe-bf0d-52c812aae1db">
</p>

Aditionally, our web-based patient monitoring & telemedicine service adopts a modular client–server architecture, emulating cloud-based solutions, for scalability and maintainability, comprising of a 1) Front-end (Client-side), 2) Back-end (Server side), 3) AI Processing Module, 4) Data Management Layer, and 5) ETL Pipeline that is accessed solely through a wireless-connection to the host device in close proximity. This enables the device to be accessed in real-time environments in remote locations for situations like: disaster relief & rescue, emergency medical treatments, and more. Moreover, we ensure compliance for mobile health platform & electronic health records (EHRs) management standards with HIPAA & GDPR for data privacy. This enables integration with large-scale healthcare services & medical teams that need a modular and mobile support tool for patient care. Below an overview of the whole system overview, interplay between modules and services, and core components for development.

<p align="center">
  <img width="700" alt="Screenshot 2025-03-18 at 4 52 41 PM" 
  src="https://github.com/user-attachments/assets/e52a32a0-94ff-4ddf-a13f-09a2b2d0ea27">
</p>

---

## 📎 **References & Further Reading**  

- 🔗 [Code](https://github.com/romerocruzsa/cp-anemia-detection/)
- 🔗 [Demo]()
- 📝 Paper: [Performance Analysis of Post-Training Quantization for CNN-based Conjunctival Pallor Anemia Detection](https://drive.google.com/file/d/1iPhDoNKccdc9iJCDJ8ZVDeg6uxaPjJdE/view?usp=share_link), International Symposium on Intelligent Computing & Networking 2025 *(Accepted)*

### Contributors  
- **Jorge A. Santiago Fernández**, jorge.santiago32@upr.edu
- **Misael J. Mercado Hernández**, misael.mercado1@upr.edu
- **Samir Y. Ali Rivera**, samir.ali@upr.edu
- **[Sebastián A. Cruz Romero](https://github.com/romerocruzsa)**, sebastian.cruz6@upr.edu
- **Dr. Wilfredo Lugo Beauchamp**, wilfredo.lugo1@upr.edu

<table align="center" style="background-color: white; border-collapse: collapse;">
  <tr>
    <td align="center" style="padding: 10px; border: none;">
      <img src="https://www.uprm.edu/wdt/resources/seal-rum-uprm-1280x1280px.png" alt="UPRM Logo" width="115">
    </td>
    <td align="center" style="padding: 10px; border: none;">
      <img src="https://www.uprm.edu/cse/wp-content/uploads/sites/153/2023/06/logo-cse-150.png" alt="CSE Department Logo" width="225">
    </td>
    <td align="center" style="padding: 10px; border: none;">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/NSF_logo.png/1019px-NSF_logo.png" alt="NSF Logo" width="115">
    </td>
    <td align="center" style="padding: 10px; border: none;">
      <img src="https://github.com/user-attachments/assets/278d7f70-5f98-45ce-a567-ac63844fe488" alt="EPSCoR Logo" width="225">
    </td>
    <td align="center" style="padding: 10px; border: none;">
      <img src="https://github.com/user-attachments/assets/d98412ac-d2d3-4bcd-96df-d73d4bfebbfb" alt="CAWT Logo" width="225">
    </td>
  </tr>
</table>

This project is supported by the University of Puerto Rico at Mayagüez Center for Research & Development, the Department of Computer Science & Engineering, and is funded by funded by the NSF-EPSCoR Center for the Advancement of Wearable Technologies (CAWT) Undergraduate Research Fall & Spring Internship 2024-25 as titled "Addressing Memory Consumption on Edge Devices through Data Quantization". (NSF Award OIA-1849243)
