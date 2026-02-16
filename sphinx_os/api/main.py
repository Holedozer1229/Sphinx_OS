"""
SphinxOS Main API

FastAPI application with all routes including treasury management
"""
from fastapi import FastAPI
from sphinx_os.api.treasury_api import router as treasury_router

app = FastAPI(
    title="SphinxSkynet API",
    description="SphinxOS API with self-funding treasury and bridge deployment",
    version="1.0.0"
)

# Include treasury routes
app.include_router(treasury_router)


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "SphinxSkynet API",
        "version": "1.0.0",
        "features": [
            "Self-Funding Treasury",
            "NFT Minting with Fees",
            "Rarity Proof System",
            "Auto Bridge Deployment"
        ]
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
