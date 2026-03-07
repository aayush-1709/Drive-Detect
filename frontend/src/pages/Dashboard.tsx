import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { UploadZone } from '../components/UploadZone';
import { ResultCard } from '../components/ResultCard';
import { 
  Car, 
  Github, 
  ShieldCheck, 
  ArrowLeft, 
  Loader2,
  History,
  ImageIcon,
  Sparkles
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { Footer } from "../components/layout/Footer";
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
  image: string;
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
  // Validate file type
const allowedTypes = ["image/jpeg", "image/png"];

if (!allowedTypes.includes(file.type)) {
  setError("Unsupported image format. Please upload JPG or PNG.");
  return;
}

// Validate file size (5MB limit)
const maxSize = 5 * 1024 * 1024;

if (file.size > maxSize) {
  setError("Image too large. Please upload an image smaller than 5MB.");
  return;
}

  const imagePreview = URL.createObjectURL(file);
setError(null);

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
  timeout: 20000
});

console.log("API response:", response.data);
      const detected = response?.data?.predictions;

if (!detected || detected.length === 0) {
  setError("No traffic sign detected. Please try another image.");
  setIsLoading(false);
  return;
}

setPredictions(detected);

// Store highest confidence prediction
if (detected.length > 0) {
  const topPrediction = detected[0];

  const newEntry = {
  label: topPrediction.class_name,
  confidence: topPrediction.confidence,
  timestamp: new Date().toLocaleString(),
  image: imagePreview,
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

  let message = "Image processing failed. Please try another image.";

  if (err?.response?.status === 413) {
    message = "Image too large. Please upload a smaller image.";
  } 
  else if (err?.response?.status === 415) {
    message = "Unsupported image format. Please upload JPG or PNG.";
  } 
 else if (err?.response?.status >= 500) {
  message = "Detection server error. The AI model may be restarting. Please try again in a moment.";
}
  else if (err?.request) {
    message = "Network error. Unable to reach the detection server.";
  }

  setError(message);
}finally {
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
        <div className="max-w-4xl mx-auto space-y-6">

  {/* Upload Section */}
  <div className="text-center">
    <h2 className="text-2xl font-semibold flex items-center justify-center gap-2 text-gray-900 dark:text-white">
      <ImageIcon size={20} />
      Upload Traffic Sign Image
    </h2>

    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
      Select or drag & drop an image to detect the traffic sign.
    </p>
  </div>

  <UploadZone onFileSelect={handleFileSelect} isLoading={isLoading} />
{isLoading && (
  <div className="mt-6 flex flex-col items-center justify-center space-y-4 text-blue-600 dark:text-blue-400">
    
    <Loader2 className="animate-spin w-10 h-10" />

    <div className="text-sm font-medium text-center space-y-1">
      <p>Uploading Image...</p>
      <p className="text-gray-500 dark:text-gray-400">
        Processing prediction with AI model
      </p>
    </div>

  </div>
)}
          
          {error && (
  <div className="mt-6 flex items-center justify-center gap-2 p-4 rounded-xl bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-800 text-red-700 dark:text-red-300 text-sm font-medium">
    <span>⚠️</span>
    <span>{error}</span>
  </div>
)}

          {predictions.length > 0 && (
  <div className="mt-8 bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-2xl p-6 shadow-sm">

    <div className="flex items-center gap-2 mb-4">
      <Sparkles size={18} className="text-blue-500" />
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
        Prediction Result
      </h3>
    </div>

    <ResultCard predictions={predictions} />

  </div>
)}
         {!isLoading && predictions.length === 0 && !error && (
  <div className="mt-6 text-center py-6 border border-dashed border-gray-300 dark:border-white/10 rounded-xl text-gray-500 dark:text-gray-400 text-sm">
    Upload an image above to start traffic sign detection.
  </div>
)}
          {/* Detection History */}
<section className="mt-16">
  <div className="flex justify-between items-center mb-4">
    <h2 className="text-xl font-semibold flex items-center gap-2 text-gray-900 dark:text-white">
  <History size={18} />
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
  <div className="text-center py-6 border border-dashed border-gray-300 dark:border-white/10 rounded-xl text-gray-500 dark:text-gray-400 text-sm">
    No detections yet. Upload an image above to start detection.
  </div>
)}

    {history.map((item, index) => (
  <div
    key={index}
    className="grid grid-cols-[60px_1fr_auto] items-center gap-4 p-4 rounded-xl bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 shadow-sm hover:shadow-md transition"
  >

    {/* Thumbnail */}
    <img
      src={item.image}
      alt="Detection"
      className="w-14 h-14 object-cover rounded-md border border-gray-200 dark:border-white/10"
    />

    {/* Detection Info */}
    <div>
      <p className="font-medium text-gray-900 dark:text-white">
        {item.label}
      </p>

      <p className="text-xs text-gray-400 mt-1">
        {item.timestamp}
      </p>
    </div>

    {/* Confidence Badge */}
    <span
      className={`text-sm font-semibold px-2 py-1 rounded-md ${
        item.confidence > 0.8
          ? "bg-green-100 text-green-600 dark:bg-green-900/30"
          : item.confidence > 0.5
          ? "bg-yellow-100 text-yellow-600 dark:bg-yellow-900/30"
          : "bg-red-100 text-red-600 dark:bg-red-900/30"
      }`}
    >
      {(item.confidence * 100).toFixed(1)}%
    </span>

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
