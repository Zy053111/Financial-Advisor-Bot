/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#0B0F19",
        card: "#151D2E",
        cardBorder: "#23304A",
      }
    },
  },
  plugins: [],
}