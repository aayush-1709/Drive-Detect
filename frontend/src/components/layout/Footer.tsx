import { Car, Github, Linkedin, X } from 'lucide-react';
import { Link } from 'react-router-dom';

const Footer = (): JSX.Element => {
  return (
    <footer className="bg-[#020202] border-t border-white/5 pt-20 pb-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Main Footer Row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-12 mb-16 text-center md:text-left">
          {/* 1. Brand — 30% */}
          <div>
            <Link
              to="/"
              className="flex items-center gap-2 mb-6 justify-center md:justify-start"
            >
              <Car size={24} className="text-blue-600" />
              <span className="font-bold text-2xl tracking-tight text-white">
                Drive<span className="text-blue-600">Detect</span>
              </span>
            </Link>

            <p className="text-gray-500 text-sm leading-relaxed max-w-sm">
              Empowering autonomous systems with real-time computer vision
              capabilities.
            </p>
          </div>

          {/* 2. Quick Links — 15% */}
          <div>
            <h3 className="font-bold text-white mb-6">Quick Links</h3>
            <ul className="space-y-4 text-sm text-gray-500">
              <li>
                <Link to="/" className="hover:text-blue-500 transition">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/about" className="hover:text-blue-500 transition">
                  About
                </Link>
              </li>
              <li>
                <Link to="/open-source" className="hover:text-blue-500 transition">
                  Open Source
                </Link>
              </li>
            </ul>
          </div>

          {/* 3. System — 15% */}
          <div>
            <h3 className="font-bold text-white mb-6">System</h3>
            <ul className="space-y-4 text-sm text-gray-500">
              <li>
                <a href="#features" className="hover:text-blue-500 transition">
                  Capabilities
                </a>
              </li>
              <li>
                <Link to="/neural-architecture" className="hover:text-blue-500 transition">
  Neural Architecture
</Link>
              </li>
              <li>
                <Link to="/app" className="hover:text-blue-500 transition">
                  Live Demo
                </Link>
              </li>
            </ul>
          </div>

          {/* 4. Support — 15% */}
          <div >
            <h3 className="font-bold text-white mb-6">Support</h3>
            <ul className="space-y-4 text-sm text-gray-500">
              <li>
                <Link to="/contact" className="hover:text-blue-500 transition">
                  Contact
                </Link>
              </li>

              {/* ✅ Feedback now goes to GitHub Issues */}
              <li>
                <Link to="/feedback" className="hover:text-blue-500 transition">
                  Feedback
                </Link>
              </li>

              <li>
                <a
                  href="https://github.com/aayush-1709/Drive-Detect/issues"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-blue-500 transition"
                >
                  Report Issue
                </a>
              </li>
            </ul>
          </div>

          {/* 5. Connect — 25% */}
          <div>
            <h3 className="font-bold text-white mb-6">Connect</h3>
            <div className="flex gap-4 justify-center md:justify-start">
              <a
                href="https://github.com/aayush-1709/Drive-Detect"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white transition"
                aria-label="GitHub"
              >
                <Github size={20} />
              </a>

              <a
                href="#"
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white transition"
                aria-label="X (Twitter)"
              >
                <X size={20} />
              </a>

              <a
                href="https://www.linkedin.com/in/aayush-sinha-481345230"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white transition"
                aria-label="LinkedIn"
              >
                <Linkedin size={20} />
              </a>
            </div>
          </div>

        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-white/5 flex flex-col md:flex-row gap-4 md:gap-0 justify-between items-center text-center md:text-left">
          <p className="text-xs text-gray-600 font-mono">
            © 2026 DRIVE DETECT / OPEN SOURCE INITIATIVE
          </p>

<div className="flex flex-col sm:flex-row gap-4 sm:gap-8 text-xs text-gray-600 font-mono">            <Link to="/privacy-policy" className="hover:text-white transition">
              PRIVACY_POLICY
            </Link>
            <Link to="/terms" className="hover:text-white transition">
              TERMS_OF_USE
            </Link>
          </div>
        </div>

      </div>
    </footer>
  );
};

export { Footer };
export default Footer;
