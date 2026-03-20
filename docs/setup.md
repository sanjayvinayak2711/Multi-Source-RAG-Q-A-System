# Setup Guide

## Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+ and pip
- **Git** for version control

## Quick Start

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd RAG-UI-System
```

### 2. Frontend Setup
```bash
cd ui
npm install
npm run dev
```
Visit: http://localhost:3000

### 3. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```
API: http://localhost:8000

### 4. Test the System
1. Open the UI in your browser
2. Upload a document (PDF, TXT, etc.)
3. Ask a question about the document
4. View the response with source citations

## Configuration

### Backend Environment
```bash
cd backend
cp .env.example .env
# Edit .env with your settings
```

### Frontend Configuration
Edit `ui/src/config.ts` for API endpoints and UI settings.

## Development Tips

- **Hot Reload**: Both frontend and backend support auto-reload
- **API Docs**: Visit http://localhost:8000/docs for Swagger UI
- **Debug Mode**: Enable DEBUG=true in .env for detailed logs

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Change ports in vite.config.ts or main.py
2. **CORS Errors**: Verify frontend URL in CORS middleware
3. **Missing Dependencies**: Run `npm install` and `pip install` again

### Getting Help
- Check the architecture documentation
- Review API endpoints at `/docs`
- Enable debug mode for detailed error logs
