import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Paper,
  Box,
  Tabs,
  Tab,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Assessment as AssessmentIcon,
  Settings as SettingsIcon,
  People as PeopleIcon,
} from '@mui/icons-material';

// Import components
import PredictionForm from './components/PredictionForm';
import ModelPerformance from './components/ModelPerformance';
import DataVisualization from './components/DataVisualization';
import BatchPrediction from './components/BatchPrediction';

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [apiStatus, setApiStatus] = useState('checking');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  // Check API status on component mount
  useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch('/health');
      if (response.ok) {
        setApiStatus('connected');
      } else {
        setApiStatus('error');
      }
    } catch (error) {
      setApiStatus('error');
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const showSnackbar = (message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <PeopleIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Customer Churn Prediction Dashboard
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2">
              API Status: 
              <Box
                component="span"
                sx={{
                  ml: 1,
                  px: 1,
                  py: 0.5,
                  borderRadius: 1,
                  backgroundColor: apiStatus === 'connected' ? 'success.main' : 'error.main',
                  color: 'white',
                  fontSize: '0.75rem',
                }}
              >
                {apiStatus === 'connected' ? 'Connected' : 'Disconnected'}
              </Box>
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
        {/* API Status Alert */}
        {apiStatus === 'error' && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            Unable to connect to the prediction API. Please ensure the backend server is running on port 8000.
          </Alert>
        )}

        {/* Tabs */}
        <Paper sx={{ mb: 3 }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            variant="fullWidth"
            indicatorColor="primary"
            textColor="primary"
          >
            <Tab
              icon={<DashboardIcon />}
              label="Single Prediction"
              iconPosition="start"
            />
            <Tab
              icon={<AssessmentIcon />}
              label="Model Performance"
              iconPosition="start"
            />
            <Tab
              icon={<PeopleIcon />}
              label="Batch Prediction"
              iconPosition="start"
            />
            <Tab
              icon={<SettingsIcon />}
              label="Data Visualization"
              iconPosition="start"
            />
          </Tabs>
        </Paper>

        {/* Tab Panels */}
        <TabPanel value={tabValue} index={0}>
          <PredictionForm onShowSnackbar={showSnackbar} />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <ModelPerformance onShowSnackbar={showSnackbar} />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <BatchPrediction onShowSnackbar={showSnackbar} />
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <DataVisualization onShowSnackbar={showSnackbar} />
        </TabPanel>
      </Container>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default App;
