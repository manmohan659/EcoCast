# 🌍 Sustainability Dashboard: Forecast & Explore Global Trends

**An interactive web application for forecasting and analyzing ecological footprints, CO₂ emissions, and sustainable growth across countries worldwide.**

---

## 🚀 Project Goal

This project enables policy makers, researchers, and the general public to intuitively explore historical data and future forecasts related to:

- **CO₂ emissions per capita**
- **Forest land productivity**
- **Ecological footprint**
- **Sustainable economic indicators**

It answers critical questions like:

- How will emissions evolve over the next decade?
- Which countries are managing ecological resources effectively?
- How do countries compare within sustainability clusters?

---

## 📂 Project Structure

Final_Project_ML/
├── artefacts/             # Clusters, model scores, manifest files
├── backend/               # FastAPI backend server
│   ├── app/
│   │   └── main.py        # API endpoints
│   └── scripts/
│       └── build_medians.py  # Data preparation scripts
├── data_work/             # Processed datasets
├── forecasts/             # Forecast CSV files (Prophet model outputs)
├── frontend/              # React.js + Vite frontend app
└── sustainability_pipeline_v2.ipynb  # Data pipeline (Jupyter notebook)

---

## 💾 Dataset Information

| Directory        | Contents                                    | Purpose             |
|------------------|---------------------------------------------|---------------------|
| `data_work/`     | Processed historical data (GDP, CO₂, footprint metrics) | Model training & clustering |
| `forecasts/`     | Country-specific 10-year forecasts           | Prophet forecasting |
| `artefacts/`     | Cluster labels, median forecasts, model scores | API consumption & comparisons |

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, Pandas
- **Frontend**: React, TypeScript, Vite, Tailwind CSS, React-Globe
- **Models**: Prophet, ARIMA, Random Forest, LightGBM (clustering & forecasting)

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Clone Repository
```bash
git clone https://docs.anthropic.com/s/claude-code-worktrees
cd claude-code-worktrees
```

### Backend Setup (FastAPI):

```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows
# .venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# Install backend dependencies
cd backend/app
pip install -r requirements.txt
cd ../..

# Run data preparation (only needed once)
python backend/scripts/build_medians.py

# Start backend server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup (React + Vite):

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

---

## 🔍 Using the Application

1. **Globe View**: Interact with the 3D globe to select countries
2. **Compare Mode**: Select multiple countries to compare sustainability metrics
3. **Forecast View**: See predicted trends for CO₂ emissions and forest productivity
4. **Country Details**: View sustainability cluster information and historical data

---

## 📊 Data Pipeline

The `sustainability_pipeline_v2.ipynb` notebook contains the full data processing workflow:
1. Data cleaning and preprocessing
2. Clustering of countries based on sustainability metrics
3. Time series forecasting using Prophet models
4. Feature importance analysis
5. HDI prediction from sustainability indicators

---

## 📝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.