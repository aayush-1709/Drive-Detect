import { useState } from "react";
import { ChevronDown } from "lucide-react";

const faqData = [
  {
    question: "What dataset is used?",
    answer:
      "DriveDetect is trained on regulatory, warning, and mandatory traffic sign datasets with augmentation techniques for robustness.",
  },
  {
    question: "What is the model accuracy?",
    answer:
      "The model achieves high classification accuracy under diverse lighting and environmental conditions.",
  },
  {
    question: "Can this be deployed locally?",
    answer:
      "Yes. DriveDetect is designed for local deployment and performs inference without requiring cloud connectivity.",
  },
  {
    question: "Is the project open source?",
    answer:
      "Yes. DriveDetect is fully open-source and available for research and educational use.",
  },
];

export function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggle = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section
      id="faq"
      className="py-24 px-6 bg-gray-50 dark:bg-gray-950 transition-colors duration-300"
    >
      <div className="max-w-4xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-12">
          Frequently <span className="text-blue-600">Asked Questions</span>
        </h2>

        <div className="space-y-4">
          {faqData.map((item, index) => (
            <div
              key={index}
              className="border border-gray-200 dark:border-white/10 rounded-xl bg-white dark:bg-white/5 transition"
            >
              <button
                onClick={() => toggle(index)}
                className="w-full flex justify-between items-center p-6 text-left"
              >
                <span className="font-medium">{item.question}</span>
                <ChevronDown
                  className={`transition-transform duration-300 ${
                    openIndex === index ? "rotate-180" : ""
                  }`}
                />
              </button>

              {openIndex === index && (
                <div className="px-6 pb-6 text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                  {item.answer}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}