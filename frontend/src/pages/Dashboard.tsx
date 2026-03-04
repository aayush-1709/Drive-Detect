import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { UploadZone } from '../components/UploadZone';
import { ResultCard } from '../components/ResultCard';
import { Car, Github, ShieldCheck, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

interface Prediction {
  class_id: number;
  class_name: string;
  confidence: number;
}

interface PredictionResponse {
  predictions: Prediction[];
}

export function Dashboard() {
  const [isLoading, setIsLoading] = useState(false);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<{
  label: string;
  confidence: number;
  timestamp: string;
}[]>([]);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const updateFavicon = () => {
      const isDark = mediaQuery.matches;
      const favicon = document.querySelector('link[rel="icon"]') as HTMLLinkElement;
      if (favicon) {
        favicon.href = isDark ? '/favicon-dark.svg' : '/favicon-light.svg';
      }
    };

    updateFavicon();
    mediaQuery.addEventListener('change', updateFavicon);
    return () => mediaQuery.removeEventListener('change', updateFavicon);
  }, []);

  useEffect(() => {
  const saved = localStorage.getItem("detectionHistory");
  if (saved) {
    setHistory(JSON.parse(saved));
  }
}, []);

useEffect(() => {
  localStorage.setItem("detectionHistory", JSON.stringify(history));
}, [history]);

  const handleFileSelect = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setPredictions([]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // In dev we can rely on Vite's proxy (`/predict` -> localhost backend).
      // In production we need an absolute backend URL unless the frontend is served by the backend.
      const base =
  import.meta.env.VITE_API_BASE ||
  'https://drive-detect-backend.onrender.com';

const url = `${base}/predict`;
      const response = await axios.post<PredictionResponse>(url, formData, {
        headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

console.log("API response:", response.data);
      const detected = response.data.predictions;
setPredictions(detected);

// Store highest confidence prediction
if (detected.length > 0) {
  const topPrediction = detected[0];

  const newEntry = {
    label: topPrediction.class_name,
    confidence: topPrediction.confidence,
    timestamp: new Date().toLocaleString(),
  };

  setHistory((prev) => {
  if (prev[0]?.label === newEntry.label &&
      prev[0]?.confidence === newEntry.confidence) {
    return prev; // prevent exact duplicate
  }
  return [newEntry, ...prev.slice(0, 9)];
});
}
    } catch (err: any) {
  console.error("API error:", err);

  if (err.response) {
    setError("Server error: Failed to process image.");
  } else if (err.request) {
    setError("Network error: Unable to reach backend service.");
  } else {
    setError("Unexpected error occurred while processing image.");
  }
}finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 font-sans selection:bg-blue-100 selection:text-blue-900">
      
      {/* Floating Navbar */}
      <header className="z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl shadow-lg mx-auto mt-0 max-w-3xl left-0 right-0 w-[95%] flex items-center justify-between px-6 h-16 sticky top-6 transition-all duration-300">
        <div className="flex items-center gap-4">
          <Link to="/" className="p-2 -ml-2 text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors rounded-full hover:bg-gray-100 dark:hover:bg-gray-800">
             <ArrowLeft size={20} />
          </Link>
          <div className="flex items-center gap-2">
            <div className="text-blue-600 dark:text-blue-400">
              <Car size={24} />
            </div>
            <span className="font-bold text-xl tracking-tight">DriveDetect</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <a href="https://github.com/aayush-1709/Drive-Detect" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors">
            <Github size={20} />
          </a>
        </div>
      </header>

      <main className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        
        {/* Header Section */}
        <div className="text-center mb-16 space-y-4">
           <div className="inline-flex items-center px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-sm font-medium mb-2">
              <ShieldCheck size={14} className="mr-1.5" />
              Live Demo
           </div>
           <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-gray-900 dark:text-white">
             Traffic Sign Recognition
           </h1>
           <p className="max-w-2xl mx-auto text-lg text-gray-600 dark:text-gray-400">
             Upload a traffic sign image to instantly classify it with our high-precision deep learning model. For test images - visit sample_images folder in the GitHub repository.
           </p>
        </div>

        {/* Interaction Area */}
        <div className="max-w-4xl mx-auto">
          <UploadZone onFileSelect={handleFileSelect} isLoading={isLoading} />
          
          {error && (
  <div className="mt-6 p-4 rounded-xl bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-800 text-red-700 dark:text-red-300 text-sm text-center font-medium">
    ❌ {error}
  </div>
)}

          <ResultCard predictions={predictions} />
          {!isLoading && predictions.length === 0 && !error && (
  <div className="mt-6 text-center text-gray-500 dark:text-gray-400 text-sm">
    Upload an image to start traffic sign detection.
  </div>
)}
          {/* Detection History */}
<section className="mt-12">
  <div className="flex justify-between items-center mb-4">
    <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
      Detection History
    </h2>

    {history.length > 0 && (
      <button
        onClick={() => setHistory([])}
        className="text-sm text-red-500 hover:text-red-600 transition"
      >
        Clear History
      </button>
    )}
  </div>

  <div className="space-y-4">
    {history.length === 0 && (
      <p className="text-sm text-gray-500 dark:text-gray-400">
        No detections yet.
      </p>
    )}

    {history.map((item, index) => (
      <div
        key={index}
        className="p-4 rounded-xl bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10"
      >
        <div className="flex justify-between items-center">
          <p className="font-medium text-gray-900 dark:text-white">
            {item.label}
          </p>

          <span
            className={`text-sm font-medium ${
              item.confidence > 0.8
                ? "text-green-500"
                : item.confidence > 0.5
                ? "text-yellow-500"
                : "text-red-500"
            }`}
          >
            {(item.confidence * 100).toFixed(2)}%
          </span>
        </div>

        <p className="text-xs text-gray-400 mt-1">
          {item.timestamp}
        </p>
      </div>
    ))}
  </div>
</section>
        </div>

      </main>

      <footer className="py-8 text-center text-sm text-gray-500 dark:text-gray-500 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
         <p>© 2026 DriveDetect. Built in Public.</p>
      </footer>
    </div>
  );
}
