import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { UploadZone } from '../components/UploadZone';
import { ResultCard } from '../components/ResultCard';
import { Car, Github, ShieldCheck, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Footer } from "../components/layout/Footer";
import { ShieldCheck } from 'lucide-react';
import { Navbar } from '../components/layout/Navbar';


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
    } catch (err) {
      console.error(err);
      setError(
        "Failed to classify image. If deployed, set VITE_API_BASE to your backend URL (e.g. https://drive-detect-backend.onrender.com)."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 font-sans selection:bg-blue-100 selection:text-blue-900">
      <Navbar />

      <main className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        
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
            <div className="mt-6 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800 text-sm text-center">
              {error}
            </div>
          )}

          <ResultCard predictions={predictions} />
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

      <Footer />
    </div>
  );
}
