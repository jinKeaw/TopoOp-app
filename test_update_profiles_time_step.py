
# Generated by Qodo Gen

from src.GUI.Dashboard_Data.Dashboard import update_profiles_time_step

import pytest

class TestUpdateProfilesTimeStep:

    # Correctly calculates time steps for valid date, hours, and minutes inputs
    def test_calculates_time_steps_correctly(self, mocker):
        from datetime import date
        import src.Environment.Env as Env
        from src.GUI.Dashboard_Data.Dashboard import update_profiles_time_step
    
        mocker.patch('src.Environment.Env.update_profiles_time_step')
        mocker.patch('src.GUI.Dashboard_Data.Dashboard.days_passed_in_year', return_value=100)
    
        chosen_date = date(2023, 4, 10)
        hours = 5
        minutes = 30
    
        expected_time_steps = ((99 * 24 * 4) + (5 * 4) + (30 / 15) + 1) - 1
    
        result = update_profiles_time_step(chosen_date, hours, minutes)
    
        assert result == expected_time_steps
        Env.update_profiles_time_step.assert_called_once_with(expected_time_steps)

    # Raises ValueError if chosen_date is not a datetime.date object
    def test_raises_value_error_for_invalid_date(self):
        from src.GUI.Dashboard_Data.Dashboard import update_profiles_time_step
    
        invalid_date = "2023-04-10"
        hours = 5
        minutes = 30
    
        with pytest.raises(ValueError, match="chosen_date must be a datetime.date object"):
            update_profiles_time_step(invalid_date, hours, minutes)