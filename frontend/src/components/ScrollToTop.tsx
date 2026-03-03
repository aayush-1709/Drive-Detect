import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { ArrowUp } from 'lucide-react';

export default function ScrollToTop() {
  const location = useLocation();
  const [isVisible, setIsVisible] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [isNearFooter, setIsNearFooter] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsVisible(window.scrollY > 300);

      const scrollTop = window.scrollY;
      const docHeight =
        document.documentElement.scrollHeight - window.innerHeight;
      const scrolled = (scrollTop / docHeight) * 100;
      setScrollProgress(scrolled);

      const footer = document.querySelector('footer');
      if (footer) {
        const footerRect = footer.getBoundingClientRect();
        setIsNearFooter(footerRect.top < window.innerHeight);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

   useEffect(() => {
    if (location.hash) {
      const id = location.hash.replace('#', '');
      const element = document.getElementById(id);
    
      if (element) {
        setTimeout(() => {
          element.scrollIntoView({ behavior: 'smooth' });
        }, 100);
      }
    }
  }, [location]);

  if (!isVisible) return null;

  const radius = 24;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset =
    circumference - (scrollProgress / 100) * circumference;

  return (
    <button
      onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
      className={`fixed right-6 md:right-8 z-50 w-14 h-14 rounded-full bg-emerald-500 text-white shadow-lg shadow-emerald-500/30 hover:bg-emerald-600 transition-all duration-300 hover:-translate-y-1 flex items-center justify-center
  ${isNearFooter ? 'bottom-24 md:bottom-16' : 'bottom-8 md:bottom-10'}`}
      aria-label="Scroll to top"
    >
      <svg
        className="absolute inset-0 rotate-[-90deg]"
        width="56"
        height="56"
        viewBox="0 0 56 56"
      >
        <circle
          cx="28"
          cy="28"
          r={radius}
          stroke="white"
          strokeWidth="4"
          fill="transparent"
          strokeOpacity="0.2"
        />
        <circle
          cx="28"
          cy="28"
          r={radius}
          stroke="white"
          strokeWidth="4"
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-200"
        />
      </svg>

      <ArrowUp size={24} className="relative z-10" />
    </button>
  );
}