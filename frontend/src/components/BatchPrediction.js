import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Button,
  Box,
  Typography,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Upload as UploadIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Send as SendIcon,
} from '@mui/icons-material';
import { apiService } from '../services/api';

const BatchPrediction = ({ onShowSnackbar }) => {
  const [customers, setCustomers] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const defaultCustomer = {
    CreditScore: '',
    Geography: '',
    Gender: '',
    Age: '',
    Tenure: '',
    Balance: '',
    NumOfProducts: '',
    HasCrCard: '',
    IsActiveMember: '',
    EstimatedSalary: '',
  };

  const addCustomer = () => {
    setCustomers([...customers, { ...defaultCustomer, id: Date.now() }]);
  };

  const removeCustomer = (id) => {
    setCustomers(customers.filter(customer => customer.id !== id));
  };

  const updateCustomer = (id, field, value) => {
    setCustomers(customers.map(customer => 
      customer.id === id ? { ...customer, [field]: value } : customer
    ));
  };

  const handleBatchPredict = async () => {
    if (customers.length === 0) {
      onShowSnackbar('Please add at least one customer', 'warning');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Convert form data to appropriate types
      const customersData = customers.map(customer => ({
        CreditScore: parseInt(customer.CreditScore),
        Geography: customer.Geography,
        Gender: customer.Gender,
        Age: parseInt(customer.Age),
        Tenure: parseInt(customer.Tenure),
        Balance: parseFloat(customer.Balance),
        NumOfProducts: parseInt(customer.NumOfProducts),
        HasCrCard: parseInt(customer.HasCrCard),
        IsActiveMember: parseInt(customer.IsActiveMember),
        EstimatedSalary: parseFloat(customer.EstimatedSalary),
      }));

      const result = await apiService.predictBatch(customersData);
      setPredictions(result.predictions);
      onShowSnackbar(`Batch prediction completed for ${customers.length} customers!`, 'success');
    } catch (err) {
      setError(err.message);
      onShowSnackbar(`Batch prediction failed: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const CHUNK_SIZE = 500; // avoid freezing the UI and backend timeouts

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        setLoading(true);
        setPredictions([]);
        const csv = e.target.result;
        const lines = csv.split(/\r?\n/);
        if (lines.length < 2) throw new Error('CSV has no data');

        const headers = lines[0].split(',').map(h => h.trim());

        // Helper to pick a field, trying several possible header names
        const pick = (rowObj, names, fallback='') => {
          for (const n of names) {
            if (rowObj[n] !== undefined && rowObj[n] !== '') return rowObj[n];
          }
          return fallback;
        };

        // Convert a CSV line into our expected customer input
        const toCustomer = (rowObj) => ({
          CreditScore: parseInt(pick(rowObj, ['CreditScore', 'credit_score']), 10),
          Geography: String(pick(rowObj, ['Geography', 'geography'])),
          Gender: String(pick(rowObj, ['Gender', 'gender'])),
          Age: parseInt(pick(rowObj, ['Age', 'age']), 10),
          Tenure: parseInt(pick(rowObj, ['Tenure', 'tenure']), 10),
          Balance: parseFloat(pick(rowObj, ['Balance', 'balance', 'AccountBalance'], 0)) || 0,
          NumOfProducts: parseInt(pick(rowObj, ['NumOfProducts', 'num_products']), 10),
          HasCrCard: parseInt(pick(rowObj, ['HasCrCard', 'has_cr_card', 'HasCreditCard'], 0), 10) || 0,
          IsActiveMember: parseInt(pick(rowObj, ['IsActiveMember', 'is_active_member'], 0), 10) || 0,
          EstimatedSalary: parseFloat(pick(rowObj, ['EstimatedSalary', 'estimated_salary'], 0)) || 0,
        });

        // Build array of row objects keyed by header
        const parsedRows = lines.slice(1)
          .filter(line => line && line.trim())
          .map((line) => {
            const values = line.split(',');
            const obj = {};
            headers.forEach((h, i) => { obj[h.trim()] = (values[i] ?? '').trim(); });
            return obj;
          });

        // Map to customer inputs alongside original IDs (if present)
        const customersWithIds = parsedRows.map((row) => ({
          sourceId: pick(row, ['id', 'ID', 'Id', 'CustomerId', 'CustomerID'], ''),
          payload: toCustomer(row),
        })).filter(({ payload }) => !Number.isNaN(payload.CreditScore) && !Number.isNaN(payload.Age));

        // Do not render all rows in the UI; process in chunks directly
        let allPreds = [];
        for (let i = 0; i < customersWithIds.length; i += CHUNK_SIZE) {
          const chunk = customersWithIds.slice(i, i + CHUNK_SIZE);
          try {
            const result = await apiService.predictBatch(chunk.map(c => c.payload));
            const preds = (result.predictions || []).map((p, idx) => ({
              ...p,
              customer_id: chunk[idx]?.sourceId || p.customer_id,
            }));
            allPreds = allPreds.concat(preds);
            onShowSnackbar(`Processed ${Math.min(i + CHUNK_SIZE, customersWithIds.length)} / ${customersWithIds.length}`, 'info');
          } catch (err) {
            onShowSnackbar(`Batch error at rows ${i+1}-${Math.min(i+CHUNK_SIZE, customersWithIds.length)}: ${err.message}`, 'error');
            break;
          }
        }

        setPredictions(allPreds);
        setCustomers([]); // keep UI light
        onShowSnackbar(`Batch prediction complete for ${allPreds.length} rows`, 'success');
      } catch (err) {
        console.error(err);
        onShowSnackbar(`Error parsing or processing CSV: ${err.message}`, 'error');
      } finally {
        setLoading(false);
      }
    };
    reader.readAsText(file);
  };

  const exportResults = () => {
    if (predictions.length === 0) {
      onShowSnackbar('No predictions to export', 'warning');
      return;
    }

    const csvContent = [
      'Customer ID,Churn Probability,Prediction,Confidence',
      ...predictions.map((pred, index) => 
        `${index + 1},${pred.churn_probability.toFixed(4)},${pred.prediction},${pred.confidence}`
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'churn_predictions.csv';
    link.click();
    window.URL.revokeObjectURL(url);
    
    onShowSnackbar('Results exported successfully!', 'success');
  };

  const getConfidenceColor = (confidence) => {
    switch (confidence) {
      case 'High': return 'success';
      case 'Medium': return 'warning';
      case 'Low': return 'error';
      default: return 'default';
    }
  };

  return (
    <Grid container spacing={3}>
      {/* Customer Input */}
      <Grid item xs={12}>
        <Card>
          <CardHeader 
            title="Customer Data Input"
            action={
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant="outlined"
                  startIcon={<UploadIcon />}
                  component="label"
                  size="small"
                >
                  Import CSV
                  <input
                    type="file"
                    accept=".csv"
                    hidden
                    onChange={handleFileUpload}
                  />
                </Button>
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={addCustomer}
                  size="small"
                >
                  Add Customer
                </Button>
              </Box>
            }
          />
          <CardContent>
            {customers.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No customers added yet
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Add customers manually or import from CSV
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={addCustomer}
                  sx={{ mt: 2 }}
                >
                  Add First Customer
                </Button>
              </Box>
            ) : (
              <Box>
                {customers.map((customer, index) => (
                  <Paper key={customer.id} sx={{ p: 2, mb: 2, border: 1, borderColor: 'grey.300' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="subtitle1">
                        Customer {index + 1}
                      </Typography>
                      <IconButton
                        onClick={() => removeCustomer(customer.id)}
                        color="error"
                        size="small"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6} md={2}>
                        <TextField
                          fullWidth
                          label="Credit Score"
                          size="small"
                          value={customer.CreditScore}
                          onChange={(e) => updateCustomer(customer.id, 'CreditScore', e.target.value)}
                          type="number"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <FormControl fullWidth size="small">
                          <InputLabel>Geography</InputLabel>
                          <Select
                            value={customer.Geography}
                            onChange={(e) => updateCustomer(customer.id, 'Geography', e.target.value)}
                            label="Geography"
                          >
                            <MenuItem value="France">France</MenuItem>
                            <MenuItem value="Germany">Germany</MenuItem>
                            <MenuItem value="Spain">Spain</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <FormControl fullWidth size="small">
                          <InputLabel>Gender</InputLabel>
                          <Select
                            value={customer.Gender}
                            onChange={(e) => updateCustomer(customer.id, 'Gender', e.target.value)}
                            label="Gender"
                          >
                            <MenuItem value="Male">Male</MenuItem>
                            <MenuItem value="Female">Female</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <TextField
                          fullWidth
                          label="Age"
                          size="small"
                          value={customer.Age}
                          onChange={(e) => updateCustomer(customer.id, 'Age', e.target.value)}
                          type="number"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <TextField
                          fullWidth
                          label="Tenure"
                          size="small"
                          value={customer.Tenure}
                          onChange={(e) => updateCustomer(customer.id, 'Tenure', e.target.value)}
                          type="number"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <TextField
                          fullWidth
                          label="Balance"
                          size="small"
                          value={customer.Balance}
                          onChange={(e) => updateCustomer(customer.id, 'Balance', e.target.value)}
                          type="number"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <TextField
                          fullWidth
                          label="Products"
                          size="small"
                          value={customer.NumOfProducts}
                          onChange={(e) => updateCustomer(customer.id, 'NumOfProducts', e.target.value)}
                          type="number"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <FormControl fullWidth size="small">
                          <InputLabel>Credit Card</InputLabel>
                          <Select
                            value={customer.HasCrCard}
                            onChange={(e) => updateCustomer(customer.id, 'HasCrCard', e.target.value)}
                            label="Credit Card"
                          >
                            <MenuItem value={1}>Yes</MenuItem>
                            <MenuItem value={0}>No</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <FormControl fullWidth size="small">
                          <InputLabel>Active Member</InputLabel>
                          <Select
                            value={customer.IsActiveMember}
                            onChange={(e) => updateCustomer(customer.id, 'IsActiveMember', e.target.value)}
                            label="Active Member"
                          >
                            <MenuItem value={1}>Yes</MenuItem>
                            <MenuItem value={0}>No</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} sm={6} md={2}>
                        <TextField
                          fullWidth
                          label="Salary"
                          size="small"
                          value={customer.EstimatedSalary}
                          onChange={(e) => updateCustomer(customer.id, 'EstimatedSalary', e.target.value)}
                          type="number"
                        />
                      </Grid>
                    </Grid>
                  </Paper>
                ))}

                {error && (
                  <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                  </Alert>
                )}

                <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                  <Button
                    variant="contained"
                    startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
                    onClick={handleBatchPredict}
                    disabled={loading || customers.length === 0}
                  >
                    {loading ? 'Predicting...' : `Predict ${customers.length} Customers`}
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<AddIcon />}
                    onClick={addCustomer}
                    disabled={loading}
                  >
                    Add Another Customer
                  </Button>
                </Box>
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>

      {/* Prediction Results */}
      {predictions.length > 0 && (
        <Grid item xs={12}>
          <Card>
            <CardHeader 
              title="Prediction Results"
              action={
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={exportResults}
                >
                  Export Results
                </Button>
              }
            />
            <CardContent>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Customer ID</TableCell>
                      <TableCell align="center">Prediction</TableCell>
                      <TableCell align="right">Churn Probability</TableCell>
                      <TableCell align="center">Confidence</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {predictions.map((prediction, index) => (
                      <TableRow key={index}>
                        <TableCell component="th" scope="row">
                          {index + 1}
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={prediction.prediction === 1 ? 'Will Churn' : 'Will Stay'}
                            color={prediction.prediction === 1 ? 'error' : 'success'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="right">
                          {(prediction.churn_probability * 100).toFixed(2)}%
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={prediction.confidence}
                            color={getConfidenceColor(prediction.confidence)}
                            size="small"
                            variant="outlined"
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      )}
    </Grid>
  );
};

export default BatchPrediction;


