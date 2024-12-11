import "./styles/App.css";

function App() {
  // Years from today to 20 years prior
  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 21 }, (_, i) => currentYear - i);

  return (
    <>
      <h1>Police Toronto - Bike Prediction App</h1>

      <div className='dashboard'>
        <div className='input-form tile'>
          <h2>Details about the stolen bike</h2>
          <form>
            {/* YEAR - OCC_YEAR */}
            <label htmlFor='year'>What year was the bike was stolen?</label>
            <select name='year' id='year'>
              <option value=''>Select Year</option>
              {/* List Options from current year to 20 years prior */}
              {years.map((year) => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </select>

            {/* MONTH - OCC_MONTH */}
            <label htmlFor='month'>What month was the bike was stolen?</label>
            <select name='month' id='month'>
              <option value=''>Select Month</option>
              <option value='January'>January</option>
              <option value='February'>February</option>
              <option value='March'>March</option>
              <option value='April'>April</option>
              <option value='May'>May</option>
              <option value='June'>June</option>
              <option value='July'>July</option>
              <option value='August'>August</option>
              <option value='September'>September</option>
              <option value='October'>October</option>
              <option value='November'>November</option>
              <option value='December'>December</option>
            </select>

            {/* DAY OF THE WEEK - OCC_DOW */}
            <label htmlFor='day'>
              What day of the week was the bike was stolen?
            </label>
            <select name='day' id='day'>
              <option value=''>Select Day</option>
              <option value='Sunday'>Sunday</option>
              <option value='Monday'>Monday</option>
              <option value='Tuesday'>Tuesday</option>
              <option value='Wednesday'>Wednesday</option>
              <option value='Thursday'>Thursday</option>
              <option value='Friday'>Friday</option>
              <option value='Saturday'>Saturday</option>
            </select>

            {/* Police Division - DIVISION */}
            <label htmlFor='division'>
              Which police division is in charge of the case?
            </label>
            <select name='division' id='division'>
              <option value=''>Select Division</option>
              {/* Display list from API */}
            </select>

            {/* Location Type - LOCATION_TYPE */}
            <label htmlFor='location_type'>
              What location type describe the best the last place your bike was
              seen?
            </label>
            <select name='location_type' id='location_type'>
              <option value=''>Select Location Type</option>
              {/* Display list from API */}
            </select>

            {/* Premises type - PREMISES_TYPE */}
            <label htmlFor='premises_type'>What premises type was it?</label>
            <select name='premises_type' id='premises_type'>
              <option value=''>Select Premises Type</option>
              {/* Display list from API */}
            </select>

            {/* Bike Cost - BIKE_COST */}
            <label htmlFor='bike_cost'>How much is the bike cost?</label>
            <input
              type='number'
              name='bike_cost'
              id='bike_cost'
              min={0}
              max={100000}
            />

            {/* Neiboorhood - NEIGHBOURHOOD_158 */}
            <label htmlFor='neighbourhood'>
              What neiboorhood was it stolen from?
            </label>
            <select name='neighbourhood' id='neighbourhood'>
              <option value=''>Select Neighbourhood</option>
              {/* Display list from API */}
            </select>

            <button type='submit'>Predict</button>
          </form>
        </div>

        <div className='prediction-results tile'>
          <h2>Outcome Predictions</h2>
          <p>Is your bike likely to be returned or not?</p>
        </div>
      </div>
    </>
  );
}

export default App;
