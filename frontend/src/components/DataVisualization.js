import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Box,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
} from 'recharts';
import { apiService } from '../services/api';

const DataVisualization = ({ onShowSnackbar }) => {
  const [featureImportance, setFeatureImportance] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedChart, setSelectedChart] = useState('feature_importance');

  useEffect(() => {
    Promise.all([fetchFeatureImportance(), fetchStats()])
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const fetchFeatureImportance = async () => {
    try {
      const data = await apiService.getFeatureImportance();
      setFeatureImportance(data);
    } catch (err) {
      setError(err.message);
      onShowSnackbar(`Failed to load visualization data: ${err.message}`, 'error');
    }
  };

  const fetchStats = async () => {
    try {
      const s = await apiService.getStats();
      setStats(s);
    } catch (err) {
      // If stats endpoint not available, keep using sample
      onShowSnackbar(`Stats endpoint error: ${err.message}`, 'warning');
    }
  };

  // Fallback demo data
  const sampleData = {
    churnByGeography: [
      { name: 'France', churned: 810, stayed: 3190, churnRate: 20.25 },
      { name: 'Germany', churned: 814, stayed: 2186, churnRate: 27.15 },
      { name: 'Spain', churned: 413, stayed: 1587, churnRate: 20.65 },
    ],
    churnByAge: [
      { age: '18-25', churned: 45, stayed: 155, churnRate: 22.5 },
      { age: '26-35', churned: 180, stayed: 720, churnRate: 20.0 },
      { age: '36-45', churned: 320, stayed: 1280, churnRate: 20.0 },
      { age: '46-55', churned: 450, stayed: 1050, churnRate: 30.0 },
      { age: '56-65', churned: 280, stayed: 420, churnRate: 40.0 },
      { age: '65+', churned: 125, stayed: 75, churnRate: 62.5 },
    ],
    churnByBalance: [
      { balance: '0', churned: 200, stayed: 800, churnRate: 20.0 },
      { balance: '1-10k', churned: 150, stayed: 850, churnRate: 15.0 },
      { balance: '10k-50k', churned: 300, stayed: 1200, churnRate: 20.0 },
      { balance: '50k-100k', churned: 400, stayed: 1000, churnRate: 28.6 },
      { balance: '100k+', churned: 350, stayed: 650, churnRate: 35.0 },
    ],
    monthlyTrends: [
      { month: 'Jan', churnRate: 18.5 },
      { month: 'Feb', churnRate: 19.2 },
      { month: 'Mar', churnRate: 20.1 },
      { month: 'Apr', churnRate: 21.3 },
      { month: 'May', churnRate: 22.0 },
      { month: 'Jun', churnRate: 20.8 },
    ],
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  const renderFeatureImportanceChart = () => {
    if (!featureImportance?.feature_importance) {
      return (
        <Alert severity="info">
          No feature importance data available.
        </Alert>
      );
    }

    const data = Object.entries(featureImportance.feature_importance)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([name, importance]) => ({
        name: name.replace(/([A-Z])/g, ' $1').trim(),
        importance: importance,
      }));

    return (
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data} layout="horizontal">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 'dataMax']} />
          <YAxis dataKey="name" type="category" width={120} />
          <Tooltip formatter={(value) => [value.toFixed(3), 'Importance']} />
          <Bar dataKey="importance" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderChurnByGeography = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={stats?.churnByGeography || sampleData.churnByGeography}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="churned" stackId="a" fill="#ff6b6b" name="Churned" />
        <Bar dataKey="stayed" stackId="a" fill="#4ecdc4" name="Stayed" />
      </BarChart>
    </ResponsiveContainer>
  );

  const renderChurnByAge = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={stats?.churnByAge || sampleData.churnByAge}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="age" />
        <YAxis />
        <Tooltip formatter={(value, name) => [value, name === 'churnRate' ? 'Churn Rate (%)' : name]} />
        <Legend />
        <Bar dataKey="churnRate" fill="#ff9ff3" name="Churn Rate (%)" />
      </BarChart>
    </ResponsiveContainer>
  );

  const renderChurnByBalance = () => (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={stats?.churnByBalance || sampleData.churnByBalance}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ name, churnRate }) => `${name}: ${churnRate}%`}
          outerRadius={120}
          fill="#8884d8"
          dataKey="churnRate"
        >
          {(stats?.churnByBalance || sampleData.churnByBalance).map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip formatter={(value) => [`${value}%`, 'Churn Rate']} />
      </PieChart>
    </ResponsiveContainer>
  );

  const renderMonthlyTrends = () => (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={sampleData.monthlyTrends}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="month" />
        <YAxis />
        <Tooltip formatter={(value) => [`${value}%`, 'Churn Rate']} />
        <Legend />
        <Line type="monotone" dataKey="churnRate" stroke="#8884d8" strokeWidth={3} name="Churn Rate (%)" />
      </LineChart>
    </ResponsiveContainer>
  );

  const renderScatterPlot = () => (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="age" name="Age" />
        <YAxis dataKey="balance" name="Balance" />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Scatter data={sampleData.churnByAge} fill="#8884d8" />
      </ScatterChart>
    </ResponsiveContainer>
  );

  const renderSelectedChart = () => {
    switch (selectedChart) {
      case 'feature_importance':
        return renderFeatureImportanceChart();
      case 'geography':
        return renderChurnByGeography();
      case 'age':
        return renderChurnByAge();
      case 'balance':
        return renderChurnByBalance();
      case 'trends':
        return renderMonthlyTrends();
      case 'scatter':
        return renderScatterPlot();
      default:
        return renderFeatureImportanceChart();
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        {error}
      </Alert>
    );
  }

  return (
    <Grid container spacing={3}>
      {/* Chart Selection */}
      <Grid item xs={12}>
        <Card>
          <CardHeader title="Data Visualizations" />
          <CardContent>
            <Box sx={{ mb: 3 }}>
              <FormControl fullWidth>
                <InputLabel>Select Visualization</InputLabel>
                <Select
                  value={selectedChart}
                  onChange={(e) => setSelectedChart(e.target.value)}
                  label="Select Visualization"
                >
                  <MenuItem value="feature_importance">Feature Importance</MenuItem>
                  <MenuItem value="geography">Churn by Geography</MenuItem>
                  <MenuItem value="age">Churn by Age Group</MenuItem>
                  <MenuItem value="balance">Churn by Balance Range</MenuItem>
                  <MenuItem value="trends">Monthly Churn Trends</MenuItem>
                  <MenuItem value="scatter">Age vs Balance Scatter</MenuItem>
                </Select>
              </FormControl>
            </Box>

            {renderSelectedChart()}
          </CardContent>
        </Card>
      </Grid>

      {/* Summary Statistics */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Churn Statistics" />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1">Overall Churn Rate</Typography>
                <Chip label={`${stats?.totals?.overallChurnRate ?? 20.37}%`} color="error" />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1">Total Customers</Typography>
                <Chip label={(stats?.totals?.totalCustomers ?? 10000).toLocaleString()} color="primary" />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1">Churned Customers</Typography>
                <Chip label={(stats?.totals?.churnedCustomers ?? 2037).toLocaleString()} color="error" />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1">Active Customers</Typography>
                <Chip label={(stats?.totals?.activeCustomers ?? 7963).toLocaleString()} color="success" />
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Key Insights */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Key Insights" />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box>
                <Typography variant="subtitle2" color="primary" gutterBottom>
                  High-Risk Segments
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Customers aged 65+ have 62.5% churn rate
                  • High balance customers (100k+) churn at 35%
                  • German customers show highest churn (27.15%)
                </Typography>
              </Box>
              <Box>
                <Typography variant="subtitle2" color="primary" gutterBottom>
                  Low-Risk Segments
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Young customers (18-25) have lower churn (22.5%)
                  • Low balance customers (1-10k) churn at 15%
                  • French customers show lowest churn (20.25%)
                </Typography>
              </Box>
              <Box>
                <Typography variant="subtitle2" color="primary" gutterBottom>
                  Recommendations
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Focus retention efforts on senior customers
                  • Implement targeted campaigns for German market
                  • Monitor high-value customers closely
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default DataVisualization;


