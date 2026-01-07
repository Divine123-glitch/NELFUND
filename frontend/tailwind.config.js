/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#e6f7e6',
          100: '#b3e6b3',
          200: '#80d580',
          300: '#4dc44d',
          400: '#1ab31a',
          500: '#008000',
          600: '#006600',
          700: '#004d00',
          800: '#003300',
          900: '#001a00',
        },
      },
    },
  },
  plugins: [],
}