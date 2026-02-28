import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Landing } from './pages/Landing';
import { Dashboard } from './pages/Dashboard';
import { About } from './pages/About';

import Contact from './pages/Contact';
import Feedback from './pages/Feedback';
import PrivacyPolicy from './pages/PrivacyPolicy';
import Terms from './pages/Terms';
import OpenSource from './pages/OpenSource';
import NotFound from './pages/NotFound';

import ScrollToTop from './components/ScrollToTop';
import NeuralArchitecture from "./pages/NeuralArchitecture";
function App(): JSX.Element {
  return (
    <>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<Dashboard />} />
        <Route path="/about" element={<About />} />

        {/* Footer pages */}
        <Route path="/contact" element={<Contact />} />
        <Route path="/feedback" element={<Feedback />} />
        <Route path="/privacy-policy" element={<PrivacyPolicy />} />
        <Route path="/terms" element={<Terms />} />
        <Route path="/open-source" element={<OpenSource />} />
        <Route path="*" element={<NotFound />} />
        <Route path="/neural-architecture" element={<NeuralArchitecture />} />
      </Routes>

      <ScrollToTop />
    </>
  );
}

export default App;