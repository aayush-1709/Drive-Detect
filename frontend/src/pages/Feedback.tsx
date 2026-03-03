import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Navbar } from "../components/layout/Navbar";

const Feedback = (): JSX.Element => {
  const [form, setForm] = useState({
    name: '',
    email: '',
    message: '',
  });

  const [errors, setErrors] = useState<{
    message?: string;
    email?: string;
  }>({});

  const [submitted, setSubmitted] = useState(false);

  const validate = () => {
    const newErrors: { message?: string; email?: string } = {};

    if (!form.message.trim()) {
      newErrors.message = 'Feedback message is required.';
    }

    if (form.email && !/^\S+@\S+\.\S+$/.test(form.email)) {
      newErrors.email = 'Please enter a valid email address.';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    setSubmitted(true);
    setForm({ name: '', email: '', message: '' });

    setTimeout(() => {
      setSubmitted(false);
    }, 3000);
  };

  return (
<div className="min-h-screen bg-white dark:bg-[#020202] text-gray-900 dark:text-white transition-colors duration-300">    
    {/* Shared Navbar */}
    <Navbar />

    <section className="px-4 sm:px-6 lg:px-8 pt-32 pb-24 flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-2xl bg-white dark:bg-[#0d0d0d] border border-gray-200 dark:border-white/10 rounded-2xl p-8 shadow-2xl transition-colors duration-300"
      >
        <h1 className="text-4xl font-bold mb-4">
          Feedback & <span className="text-blue-600">Suggestions</span>
        </h1>

        <p className="text-gray-600 dark:text-gray-400 mb-8 leading-relaxed">
          Your feedback helps improve DriveDetect. Whether it’s UI/UX,
          performance, documentation, or new ideas — every suggestion matters.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">

          {/* Name */}
          <input
            type="text"
            placeholder="Your Name (optional)"
            value={form.name}
            onChange={(e) =>
              setForm({ ...form, name: e.target.value })
            }
            className="w-full px-4 py-3 rounded-xl bg-gray-100 dark:bg-white/5 border border-gray-300 dark:border-white/10 text-gray-900 dark:text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-600 transition"
          />

          {/* Email */}
          <div>
            <input
              type="email"
              placeholder="Your Email (optional)"
              value={form.email}
              onChange={(e) =>
                setForm({ ...form, email: e.target.value })
              }
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-600 transition"
            />
            {errors.email && (
              <p className="text-red-500 text-sm mt-2">
                {errors.email}
              </p>
            )}
          </div>

          {/* Message */}
          <div>
            <textarea
              rows={4}
              placeholder="Your Feedback"
              value={form.message}
              onChange={(e) =>
                setForm({ ...form, message: e.target.value })
              }
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-600 transition resize-none"
            />
            {errors.message && (
              <p className="text-red-500 text-sm mt-2">
                {errors.message}
              </p>
            )}
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold shadow-lg shadow-blue-600/20 transition-all active:scale-95"
          >
            Submit Feedback
          </button>
        </form>

        {/* Success Message */}
        <AnimatePresence>
          {submitted && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="mt-6 p-4 rounded-xl bg-green-600/20 border border-green-600/30 text-green-400 text-center"
            >
              🎉 Thank you for your feedback!
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </section>
  </div>
);
};

export default Feedback;